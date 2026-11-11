"""End-to-end test of match recording + review against a real pFMS harness.

Counterpart to pFMS's scripts/e2e-match-harness.ts (start that first).
Runs the REAL PfmsMatchLink, MatchRecorder, and web app — only the camera is
replaced by a synthetic 30 fps frame feeder. Exercises: recording driven by
the harness's WebSocket match state (including a mid-teleop pause/resume),
finalization, review-page APIs, mark tallying (pause marks excluded), and
score report-back to the harness.

Run: uv run --extra web python scripts/e2e_match_review.py
"""

import json
import shutil
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from ball_counter.buffer import BufferFrame
from ball_counter.config import PfmsConfig
from ball_counter.match import MatchRecorder, PfmsMatchLink, load_match, tally_marks
from ball_counter.web import AppState, start_server_thread

PFMS_URL = "http://127.0.0.1:39871"
WEB_PORT = 39872
WEB = f"http://127.0.0.1:{WEB_PORT}"
GOALS = ("red-goal", "blue-goal")


def api(method: str, path: str, body: dict | None = None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        WEB + path, data=data, method=method,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())


def fake_frame(idx: int) -> BufferFrame:
    img = np.full((60, 80, 3), (idx * 3) % 255, dtype=np.uint8)
    ok, jpeg = cv2.imencode(".jpg", img)
    assert ok
    return BufferFrame(
        timestamp=time.strftime("%H:%M:%S"), jpeg=jpeg.tobytes(),
        frame_idx=idx, signal=0, rising=False,
    )


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="balls-e2e-"))
    clips_dir = tmp / "clips"
    matches_dir = clips_dir / "matches"
    clips_dir.mkdir(parents=True)

    cfg = PfmsConfig(url=PFMS_URL, key=None, source="e2e", public_url=WEB)
    recorder = MatchRecorder(matches_dir, fps=30.0)
    link = PfmsMatchLink(cfg, recorder)
    link.start()

    state = AppState()
    state.set_clips_dir(clips_dir)
    state.set_match_context(matches_dir, recorder, link)
    start_server_thread(state, port=WEB_PORT, host="127.0.0.1")

    # Synthetic camera: feed both goals at ~30 fps while a match records
    stop = threading.Event()

    def feeder():
        idx = 0
        while not stop.is_set():
            if recorder.active:
                for goal in GOALS:
                    recorder.on_frame(goal, fake_frame(idx))
                idx += 1
            time.sleep(1 / 30)

    threading.Thread(target=feeder, daemon=True).start()

    try:
        print("e2e: waiting for the harness to start a match...")
        deadline = time.time() + 120
        while not recorder.active:
            if time.time() > deadline:
                print("E2E RESULT: FAIL — no match started (is the harness running?)")
                return 1
            time.sleep(0.5)
        match_id = recorder.match_id
        print(f"e2e: recording match {match_id}")

        print("e2e: waiting for the match to end and finalize...")
        deadline = time.time() + 300
        while load_match(matches_dir, match_id) is None:
            if time.time() > deadline:
                print("E2E RESULT: FAIL — match never finalized")
                return 1
            time.sleep(1)
        print("e2e: finalized")

        detail = api("GET", f"/api/match/{match_id}")
        failures: list[str] = []

        phases = [ev["phase"] for ev in detail["timeline"]]
        print(f"e2e: timeline phases: {phases}")
        for expected in ("countdown", "auto", "autoPause", "teleop", "paused", "endgame", "postMatch"):
            if expected not in phases:
                failures.append(f"phase {expected} missing from timeline")

        # Shift sub-periods (harness plays transition, jumps to shift4, endgame)
        subs = [ev.get("sub") for ev in detail["timeline"]]
        print(f"e2e: timeline subs: {subs}")
        for expected_sub in ("transition", "shift4", "endgame"):
            if expected_sub not in subs:
                failures.append(f"sub-period {expected_sub} missing from timeline")

        # Exactly one goal (the auto loser's) has an inactive shift-4 span;
        # a mark deep in it must not count, one within the 3s grace must.
        inactive_goals = [g for g in GOALS
                          if any(s["phase"] == "teleop" and s.get("active") is False
                                 for s in detail["goals"][g]["spans"])]
        print(f"e2e: goals with an inactive shift span: {inactive_goals}")
        if len(inactive_goals) != 1:
            failures.append(f"expected exactly 1 goal with an inactive span, got {inactive_goals}")
        else:
            g = inactive_goals[0]
            spans = detail["goals"][g]["spans"]
            s = next(s for s in spans if s["phase"] == "teleop" and s.get("active") is False)
            if s["end"] - s["start"] < 4.0:
                failures.append(f"inactive span too short to test grace: {s}")
            else:
                t = tally_marks(
                    [{"video_time": s["start"] + 3.5, "n_balls": 1},   # hub off > grace
                     {"video_time": s["start"] + 1.0, "n_balls": 1}],  # ball in flight (grace)
                    spans,
                )
                print(f"e2e: inactive-span tally on {g}: {t}")
                if not (t["score"] == 1 and t["goalInactive"] == 1):
                    failures.append(f"inactive/grace tally wrong: {t}")
        if set(detail["goals"].keys()) != set(GOALS):
            failures.append(f"unexpected goals: {list(detail['goals'])}")
        teams = {t["alliance"]: t["team"] for t in detail.get("teams", [])}
        if teams != {"red": 1234, "blue": 5678}:
            failures.append(f"unexpected teams: {teams}")

        def span(goal: str, phase: str, last: bool = False) -> dict:
            spans = [s for s in detail["goals"][goal]["spans"] if s["phase"] == phase]
            if not spans:
                raise AssertionError(f"no {phase} span for {goal}")
            return spans[-1 if last else 0]

        def mid(s: dict) -> float:
            return (s["start"] + s["end"]) / 2

        # Red reviewer: 1 ball in auto, 2 in teleop, 1 during the operator
        # pause (must be excluded) -> reported score 3, auto 1
        red = api("POST", "/api/reviewer/create", {"label": "Red Reviewer E2E"})
        marks = [
            {"video_time": mid(span("red-goal", "auto")), "n_balls": 1},
            {"video_time": mid(span("red-goal", "teleop")), "n_balls": 2},
            {"video_time": mid(span("red-goal", "paused")), "n_balls": 1},
        ]
        api("POST", f"/api/match/{match_id}/red-goal/annotations",
            {"token": red["token"], "label": red["label"], "marks": marks})
        r = api("POST", f"/api/match/{match_id}/submit", {"token": red["token"], "alliance": "red"})
        print(f"e2e: red submit -> {r['submitted']}")
        if not (r["ok"] and r["submitted"]["score"] == 3 and r["submitted"]["autoScore"] == 1
                and r["submitted"]["excluded"] == 1):
            failures.append(f"red tally wrong: {r['submitted']}")

        # Blue reviewer: 2 balls in endgame -> score 2, auto 0
        blue = api("POST", "/api/reviewer/create", {"label": "Blue Reviewer E2E"})
        marks = [{"video_time": mid(span("blue-goal", "endgame")), "n_balls": 2}]
        api("POST", f"/api/match/{match_id}/blue-goal/annotations",
            {"token": blue["token"], "label": blue["label"], "marks": marks})
        r = api("POST", f"/api/match/{match_id}/submit", {"token": blue["token"], "alliance": "blue"})
        print(f"e2e: blue submit -> {r['submitted']}")
        if not (r["ok"] and r["submitted"]["score"] == 2 and r["submitted"]["autoScore"] == 0):
            failures.append(f"blue tally wrong: {r['submitted']}")

        # Videos exist and are non-trivial
        for goal in GOALS:
            mp4 = matches_dir / f"{match_id}_{goal}.mp4"
            if not mp4.exists() or mp4.stat().st_size < 10_000:
                failures.append(f"video missing/too small: {mp4.name}")

        if failures:
            print("E2E RESULT: FAIL —", "; ".join(failures))
            return 1
        print("E2E RESULT: PASS — recorded, reviewed, and reported back to pFMS")
        return 0
    finally:
        stop.set()
        time.sleep(0.2)
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
