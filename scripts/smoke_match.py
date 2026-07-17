"""Smoke test for match.py: recorder lifecycle, spans, and tally logic.

Run: uv run --extra web python scripts/smoke_match.py
Simulates a short match with fake frames (no PFMS, no camera).
"""

import json
import shutil
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

from ball_counter.buffer import BufferFrame
from ball_counter.match import (
    MatchRecorder,
    goal_spans,
    load_match,
    tally_marks,
)


def fake_frame(idx: int) -> BufferFrame:
    img = np.full((60, 80, 3), (idx * 7) % 255, dtype=np.uint8)
    ok, jpeg = cv2.imencode(".jpg", img)
    assert ok
    return BufferFrame(
        timestamp=f"00:00:{idx:02d}", jpeg=jpeg.tobytes(), frame_idx=idx,
        signal=idx % 50, rising=bool(idx % 2),
    )


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="match-smoke-"))
    try:
        rec = MatchRecorder(tmp, fps=30.0)
        assert not rec.active

        rec.begin("test-match-1", {"matchNumber": 7, "teams": [], "config": {}}, "countdown")
        assert rec.active and rec.match_id == "test-match-1"

        # countdown: 15 frames (0.5s), auto: 30, paused: 30, teleop: 30, postMatch: 15
        idx = 0
        for phase, n in [("countdown", 15), ("auto", 30), ("paused", 30), ("teleop", 30), ("postMatch", 15)]:
            if phase != "countdown":
                rec.add_event(phase)
            for _ in range(n):
                rec.on_frame("red-goal", fake_frame(idx))
                idx += 1

        # A live mark during teleop (frame ~90 => t=3.0s)
        vt = rec.add_live_mark("tok1", "Tester", "red-goal", 1)
        assert vt is not None and abs(vt - 120 / 30.0) < 0.2, vt

        rec.finish("normal", tail_sec=0.0)
        for _ in range(100):
            if not rec.active and load_match(tmp, "test-match-1"):
                break
            time.sleep(0.1)

        meta = load_match(tmp, "test-match-1")
        assert meta, "match json not written"
        assert meta["end_reason"] == "normal"
        assert meta["goals"]["red-goal"]["n_frames"] == 120, meta["goals"]

        spans = goal_spans(meta, "red-goal")
        print("spans:", json.dumps(spans, indent=1))
        phases = [s["phase"] for s in spans]
        assert phases == ["countdown", "auto", "paused", "teleop", "postMatch"], phases
        assert not spans[0]["scoring"] and spans[1]["scoring"] and not spans[2]["scoring"]

        # Tally: marks in countdown (excluded), auto, paused (excluded), teleop x2
        marks = [
            {"video_time": 0.2, "n_balls": 1},   # countdown -> excluded
            {"video_time": 0.7, "n_balls": 2},   # auto -> auto portion
            {"video_time": 1.6, "n_balls": 1},   # paused -> excluded
            {"video_time": 2.6, "n_balls": 3},   # teleop
        ]
        t = tally_marks(marks, spans)
        assert t == {"score": 5, "autoScore": 2, "excluded": 2}, t

        # Live marks persisted into the sidecar
        sidecar = json.loads((tmp / "test-match-1_red-goal.json").read_text())
        assert "tok1" in sidecar["annotations"], sidecar.get("annotations")
        assert (tmp / "test-match-1_red-goal.mp4").exists()

        # Abort path: nothing left behind
        rec2 = MatchRecorder(tmp, fps=30.0)
        rec2.begin("test-match-2", {}, "countdown")
        rec2.on_frame("red-goal", fake_frame(0))
        rec2.abort()
        assert not rec2.active
        assert load_match(tmp, "test-match-2") is None

        print("smoke_match: ALL OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
