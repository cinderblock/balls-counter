"""Match-scoped recording and score review, driven by PFMS match state.

PFMS broadcasts its match state on a public read-only WebSocket
(``/ws/scores``).  :class:`PfmsMatchLink` subscribes to it and drives a
:class:`MatchRecorder`: one continuous recording per goal spanning the whole
match — countdown pre-roll, auto, the auto→teleop pause, teleop, endgame, any
operator pauses, and the post-match ball-count period — with the phase
timeline captured per goal-video so reviewers can see exactly which spans
count toward the score.

Recordings land in ``<clips_dir>/matches/`` as::

    <match_id>_<goal>.mp4 / .json   # same sidecar shape as regular clips
    <match_id>.match.json           # match metadata + phase timeline

When a recording is finalized, the review page URL is registered back with
PFMS (``POST /api/match-review/recording``) so the match history page links
straight to it.  Reviewed scores go back via ``POST /api/match-review`` on
the same channel the live score forwarder already uses.
"""

import json
import logging
import re
import tempfile
import threading
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from ball_counter.buffer import BufferFrame
from ball_counter.clips import _reencode_h264
from ball_counter.config import PfmsConfig

log = logging.getLogger(__name__)

# Phases during which balls count toward the score.  autoPause and postMatch
# are included: balls launched before the buzzer are still in flight then.
SCORING_PHASES = {"auto", "autoPause", "teleop", "endgame", "postMatch"}
# Phases during which the auto portion of the score accumulates.
AUTO_PHASES = {"auto", "autoPause"}
# Phases that mean a match is underway (recording should be running).
ACTIVE_PHASES = {"countdown", "auto", "autoPause", "paused", "teleop", "endgame"}

# Keep recording this long into postMatch so balls in flight at the final
# buzzer (up to 3 s per the rules) are on video, with a little slack.
POST_MATCH_TAIL_SEC = 6.0

# Safety cap on a single match recording (a normal match is ~3 minutes; this
# allows long operator pauses but stops a runaway recording if the match-end
# signal is never seen).
MAX_MATCH_SEC = 30 * 60

# If the PFMS connection stays down this long during a match, assume the
# match is over and finalize what we have.
PFMS_LOST_GRACE_SEC = 120.0


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


class _GoalRecording:
    """Incremental MP4 + sidecar rows for a single goal during one match."""

    def __init__(self, fps: float):
        self.fps = fps
        self.writer: cv2.VideoWriter | None = None
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            self.tmp_path = Path(tmp.name)
        self.rows: list[dict] = []
        self.n_frames = 0
        self.last_frame_idx: int | None = None

    def write(self, bf: BufferFrame) -> None:
        if bf.frame_idx == self.last_frame_idx:
            return  # same frame delivered twice
        self.last_frame_idx = bf.frame_idx
        arr = np.frombuffer(bf.jpeg, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return
        if self.writer is None:
            h, w = img.shape[:2]
            self.writer = cv2.VideoWriter(
                str(self.tmp_path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h)
            )
        self.writer.write(img)
        ev = None
        if bf.event is not None:
            ev = {
                "frame": bf.event.frame,
                "n_balls": bf.event.n_balls,
                "peak_area": bf.event.peak_area,
            }
        self.rows.append({
            "frame_idx": self.n_frames,
            "timestamp": bf.timestamp,
            "signal": bf.signal,
            "rising": bf.rising,
            "event": ev,
        })
        self.n_frames += 1

    def close(self) -> None:
        if self.writer is not None:
            self.writer.release()
            self.writer = None

    def discard(self) -> None:
        self.close()
        self.tmp_path.unlink(missing_ok=True)


class MatchRecorder:
    """Records every goal continuously for the duration of one PFMS match.

    ``on_frame`` is called from the main processing loop; all control methods
    are called from the PFMS WebSocket thread or web request threads.
    """

    def __init__(self, matches_dir: Path, fps: float = 30.0):
        self._matches_dir = matches_dir
        self._fps = fps
        self._lock = threading.Lock()
        self._goals: dict[str, _GoalRecording] = {}
        self._meta: dict | None = None
        self._match_id: str | None = None
        self._accepting = False  # frames still being written
        self._finalize_timer: threading.Timer | None = None
        # Marks placed live (while recording): goal -> token -> {label, marks}
        self._live_marks: dict[str, dict[str, dict]] = {}
        # Called with the finished match metadata dict after finalize
        self.on_finalized = None

    # ── State ────────────────────────────────────────────────────────

    @property
    def active(self) -> bool:
        """True while a match recording is accepting frames."""
        with self._lock:
            return self._accepting

    @property
    def match_id(self) -> str | None:
        with self._lock:
            return self._match_id

    def live_positions(self) -> dict[str, float]:
        """Current video time (seconds) per goal — for live marking."""
        with self._lock:
            return {name: g.n_frames / self._fps for name, g in self._goals.items()}

    def live_summary(self) -> dict | None:
        """Snapshot of the in-progress match in the same shape as .match.json."""
        with self._lock:
            if not self._accepting or self._meta is None:
                return None
            m = self._meta
            return {
                "match_id": m["match_id"],
                "match_number": m.get("match_number"),
                "started_at": m["started_at"],
                "started_ts": m["started_ts"],
                "late_start": m["late_start"],
                "teams": list(m["teams"]),
                "config": dict(m["config"]),
                "timeline": [dict(ev) for ev in m["timeline"]],
                "end_reason": m["end_reason"],
                "goals": {
                    name: {"clip": f"{m['match_id']}_{name}", "fps": self._fps, "n_frames": g.n_frames}
                    for name, g in self._goals.items()
                },
                "submitted": {},
            }

    # ── Control (PFMS link thread) ───────────────────────────────────

    def begin(self, match_id: str, meta: dict, phase: str, late: bool = False) -> None:
        """Start recording all goals for a new match."""
        with self._lock:
            if self._accepting:
                return
            self._match_id = match_id
            self._goals = {}
            self._live_marks = {}
            self._meta = {
                "match_id": match_id,
                "match_number": meta.get("matchNumber"),
                "started_at": _now_iso(),
                "started_ts": time.time(),
                "late_start": late,
                "teams": meta.get("teams", []),
                "config": meta.get("config", {}),
                "timeline": [],
                "end_reason": None,
                "goals": {},
                "submitted": {},
            }
            self._accepting = True
        self.add_event(phase)
        log.info("match %s: recording started (phase %s%s)", match_id, phase, ", late" if late else "")
        print(f"match    - {match_id[:8]}: recording started ({phase}{', late join' if late else ''})")

    def add_event(self, phase: str) -> None:
        """Append a phase-timeline entry, pinned to each goal's video position."""
        with self._lock:
            if self._meta is None or not self._accepting:
                return
            self._meta["timeline"].append({
                "phase": phase,
                "ts": time.time(),
                "frame_pos": {name: g.n_frames for name, g in self._goals.items()},
            })
        print(f"match    - phase: {phase}")

    def add_live_mark(self, token: str, label: str, goal: str, n_balls: int = 1) -> float | None:
        """Place a reviewer mark at the current video position while recording.

        Returns the mark's video time, or None if not recording that goal yet.
        """
        with self._lock:
            if not self._accepting:
                return None
            rec = self._goals.get(goal)
            if rec is None:
                return None
            video_time = rec.n_frames / self._fps
            anno = self._live_marks.setdefault(goal, {}).setdefault(
                token, {"label": label, "marks": []}
            )
            anno["marks"].append({
                "video_time": video_time,
                "frame_idx": rec.n_frames,
                "timestamp": _now_iso(),
                "n_balls": n_balls,
            })
            return video_time

    def abort(self) -> None:
        """Countdown was aborted — the match never happened; discard everything."""
        with self._lock:
            if not self._accepting:
                return
            self._cancel_timer()
            goals = self._goals
            match_id = self._match_id
            self._goals = {}
            self._meta = None
            self._match_id = None
            self._accepting = False
        for g in goals.values():
            g.discard()
        log.info("match %s: recording discarded (countdown aborted)", match_id)
        print(f"match    - {str(match_id)[:8]}: discarded (countdown aborted)")

    def finish(self, end_reason: str, tail_sec: float = POST_MATCH_TAIL_SEC) -> None:
        """Stop recording after ``tail_sec`` more video, then finalize on a worker thread."""
        with self._lock:
            if self._meta is None or self._finalize_timer is not None:
                return
            self._meta["end_reason"] = end_reason
            timer = threading.Timer(max(0.0, tail_sec), self._finalize)
            timer.daemon = True
            self._finalize_timer = timer
        timer.start()
        print(f"match    - ending ({end_reason}), finalizing in {tail_sec:.0f}s")

    # ── Frame feed (main processing loop) ────────────────────────────

    def on_frame(self, goal_name: str, bf: BufferFrame) -> None:
        with self._lock:
            if not self._accepting:
                return
            rec = self._goals.get(goal_name)
            if rec is None:
                rec = _GoalRecording(self._fps)
                self._goals[goal_name] = rec
        try:
            rec.write(bf)
        except Exception as exc:
            log.warning("match frame write failed for %s: %s", goal_name, exc)
            return
        if rec.n_frames > self._fps * MAX_MATCH_SEC:
            self.finish("timeout", tail_sec=0.0)

    # ── Finalize ─────────────────────────────────────────────────────

    def _cancel_timer(self) -> None:
        if self._finalize_timer is not None:
            self._finalize_timer.cancel()
            self._finalize_timer = None

    def _finalize(self) -> None:
        with self._lock:
            self._accepting = False
            self._finalize_timer = None
            goals = self._goals
            meta = self._meta
            live_marks = self._live_marks
            match_id = self._match_id
            self._goals = {}
            self._meta = None
            self._match_id = None
            self._live_marks = {}
        if meta is None or match_id is None:
            return

        self._matches_dir.mkdir(parents=True, exist_ok=True)

        # If this match was already finalized once (recording restarted after a
        # long PFMS outage), save this segment under a -pN suffix instead of
        # overwriting the earlier files.
        base_id = match_id
        part = 2
        while (self._matches_dir / f"{match_id}.match.json").exists():
            match_id = f"{base_id}-p{part}"
            part += 1
        if match_id != base_id:
            meta["continuation_of"] = base_id
            meta["match_id"] = match_id
        for name, rec in goals.items():
            rec.close()
            if rec.n_frames == 0:
                rec.discard()
                continue
            stem = f"{match_id}_{name}"
            mp4_path = self._matches_dir / f"{stem}.mp4"
            json_path = self._matches_dir / f"{stem}.json"
            try:
                _reencode_h264(rec.tmp_path, mp4_path, self._fps)
            except Exception as exc:
                log.error("match %s: re-encode failed for %s: %s", match_id, name, exc)
                rec.discard()
                continue
            rec.tmp_path.unlink(missing_ok=True)
            sidecar = {
                "goal": name,
                "match_id": match_id,
                "saved_at": _now_iso(),
                "fps": self._fps,
                "n_frames": rec.n_frames,
                "frames": rec.rows,
            }
            marks = live_marks.get(name)
            if marks:
                sidecar["annotations"] = {
                    token: {"label": a["label"], "saved_at": _now_iso(), "marks": a["marks"]}
                    for token, a in marks.items()
                }
            json_path.write_text(json.dumps(sidecar, indent=2))
            meta["goals"][name] = {"clip": stem, "fps": self._fps, "n_frames": rec.n_frames}
            print(f"match    - saved {mp4_path.name} ({rec.n_frames / self._fps:.0f}s)")

        meta["finalized_at"] = _now_iso()
        (self._matches_dir / f"{match_id}.match.json").write_text(json.dumps(meta, indent=2))
        print(f"match    - {match_id[:8]}: finalized ({len(meta['goals'])} goal(s))")

        if self.on_finalized is not None:
            try:
                self.on_finalized(meta)
            except Exception as exc:
                log.warning("match on_finalized callback failed: %s", exc)


# ── Match store helpers (used by the web layer) ──────────────────────


def load_match(matches_dir: Path, match_id: str) -> dict | None:
    if not re.fullmatch(r"[A-Za-z0-9-]+", match_id):
        return None
    path = matches_dir / f"{match_id}.match.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def save_match(matches_dir: Path, meta: dict) -> None:
    (matches_dir / f"{meta['match_id']}.match.json").write_text(json.dumps(meta, indent=2))


def list_matches(matches_dir: Path) -> list[dict]:
    """All finalized matches, newest first."""
    if not matches_dir.exists():
        return []
    matches = []
    for path in matches_dir.glob("*.match.json"):
        try:
            matches.append(json.loads(path.read_text()))
        except Exception:
            continue
    matches.sort(key=lambda m: m.get("started_ts", 0), reverse=True)
    return matches


def goal_spans(meta: dict, goal: str) -> list[dict]:
    """The phase timeline as video-time spans for one goal's recording.

    Returns ``[{phase, start, end, scoring}]`` in seconds of that goal's video.
    Each timeline event was pinned to the goal's frame count when it happened,
    so spans stay exact even when the camera dropped frames.
    """
    info = meta.get("goals", {}).get(goal)
    if info is None:
        return []
    fps = info.get("fps") or 30.0
    total = (info.get("n_frames") or 0) / fps
    timeline = meta.get("timeline", [])
    spans = []
    for i, ev in enumerate(timeline):
        start = ev.get("frame_pos", {}).get(goal, 0) / fps
        if i + 1 < len(timeline):
            end = timeline[i + 1].get("frame_pos", {}).get(goal, 0) / fps
        else:
            end = total
        phase = ev.get("phase", "")
        spans.append({
            "phase": phase,
            "start": round(start, 3),
            "end": round(end, 3),
            "scoring": phase in SCORING_PHASES,
        })
    return spans


def tally_marks(marks: list[dict], spans: list[dict]) -> dict:
    """Sum a reviewer's marks over the scoring spans.

    Returns ``{"score": total, "autoScore": auto_portion, "excluded": n}`` —
    marks placed outside scoring spans (countdown, operator pauses) are not
    counted and reported as excluded.
    """
    score = 0
    auto = 0
    excluded = 0
    for m in marks:
        t = m.get("video_time", 0)
        n = int(m.get("n_balls", 1))
        span = next((s for s in spans if s["start"] <= t < s["end"]), None)
        if span is None or not span["scoring"]:
            excluded += n
            continue
        score += n
        if span["phase"] in AUTO_PHASES:
            auto += n
    return {"score": score, "autoScore": auto, "excluded": excluded}


def alliance_for_goal(goal: str) -> str | None:
    return "red" if "red" in goal else "blue" if "blue" in goal else None


# ── PFMS link ────────────────────────────────────────────────────────


class PfmsMatchLink:
    """Subscribes to the PFMS match-state WebSocket and drives the recorder.

    Also carries reviewed scores and recording registrations back to PFMS
    over the same HTTP channel the live score forwarder uses.
    """

    def __init__(self, pfms: PfmsConfig, recorder: MatchRecorder):
        self._pfms = pfms
        self._recorder = recorder
        self._ws_url = re.sub(r"^http", "ws", pfms.url.rstrip("/")) + "/ws/scores"
        self._headers = {"Content-Type": "application/json"}
        if pfms.key:
            self._headers["X-API-Key"] = pfms.key
        self._prev_phase: str | None = None
        recorder.on_finalized = self._register_recording

    def start(self) -> None:
        t = threading.Thread(target=self._run, daemon=True, name="pfms-match-link")
        t.start()

    # ── WebSocket loop ───────────────────────────────────────────────

    def _run(self) -> None:
        try:
            from websockets.sync.client import connect
        except ImportError:
            print("match    - DISABLED: 'websockets' package not installed (install the [web] extra)")
            return

        announced_fail = False
        lost_since: float | None = None
        while True:
            try:
                with connect(self._ws_url, open_timeout=5, close_timeout=1) as ws:
                    print(f"match    - connected to PFMS at {self._ws_url}")
                    announced_fail = False
                    lost_since = None
                    self._prev_phase = None
                    while True:
                        # The client's ping/pong keepalive detects dead
                        # connections; recv can block indefinitely.
                        msg = ws.recv()
                        try:
                            data = json.loads(msg)
                        except (TypeError, ValueError):
                            continue
                        if isinstance(data, dict) and data.get("type") == "matchState":
                            self._handle(data)
            except Exception as exc:
                if not announced_fail:
                    print(f"match    - PFMS connection lost ({exc}), retrying")
                    announced_fail = True
                # A brief blip mid-match is fine — the recording continues and
                # we pick the state back up on reconnect.  Only give up on the
                # match if PFMS stays gone.
                if self._recorder.active:
                    if lost_since is None:
                        lost_since = time.monotonic()
                    elif time.monotonic() - lost_since > PFMS_LOST_GRACE_SEC:
                        self._recorder.finish("pfms-lost", tail_sec=0.0)
                        lost_since = None
                time.sleep(2)

    def _handle(self, state: dict) -> None:
        phase = state.get("phase")
        match_id = state.get("matchId")
        prev = self._prev_phase
        if phase == prev:
            return
        self._prev_phase = phase

        recorder = self._recorder

        if recorder.active and match_id and recorder.match_id not in (None, match_id):
            # A different match started while we were still recording the old
            # one (missed transitions) — close out the old recording first.
            recorder.finish("superseded", tail_sec=0.0)

        if not recorder.active:
            if match_id and phase in ACTIVE_PHASES:
                meta = self._extract_meta(state)
                recorder.begin(match_id, meta, phase, late=(phase != "countdown"))
            return

        # Recording is active
        if phase == "created" and prev == "countdown":
            recorder.abort()
        elif phase in ACTIVE_PHASES:
            recorder.add_event(phase)
        elif phase == "postMatch":
            recorder.add_event("postMatch")
            recorder.finish(state.get("endReason") or "normal")
        elif phase in ("idle", "created"):
            # Abandoned from pause (or transitions missed) — no postMatch came
            recorder.add_event(phase)
            recorder.finish(state.get("endReason") or "abandoned", tail_sec=0.0)

    @staticmethod
    def _extract_meta(state: dict) -> dict:
        teams = []
        for station, ss in (state.get("stationStates") or {}).items():
            if not isinstance(ss, dict) or not ss.get("joined"):
                continue
            teams.append({
                "team": ss.get("teamNumber"),
                "alliance": ss.get("alliance"),
                "slot": ss.get("matchSlot"),
                "station": station,
            })
        return {
            "matchNumber": state.get("matchNumber"),
            "teams": teams,
            "config": state.get("config") or {},
        }

    # ── HTTP back-channel to PFMS ────────────────────────────────────

    def _post(self, path: str, payload: dict, attempts: int = 3) -> tuple[bool, str]:
        url = self._pfms.url.rstrip("/") + path
        data = json.dumps(payload).encode()
        last_err = ""
        for i in range(attempts):
            try:
                req = urllib.request.Request(url, data=data, headers=self._headers, method="POST")
                with urllib.request.urlopen(req, timeout=5) as resp:
                    json.loads(resp.read())
                return True, ""
            except Exception as exc:
                last_err = str(exc)
                if i + 1 < attempts:
                    time.sleep(5)
        log.warning("PFMS POST %s failed: %s", path, last_err)
        return False, last_err

    def _register_recording(self, meta: dict) -> None:
        if not self._pfms.public_url:
            return
        url = f"{self._pfms.public_url.rstrip('/')}/match/{meta['match_id']}"
        ok, err = self._post(
            "/api/match-review/recording", {"matchId": meta["match_id"], "url": url}
        )
        if ok:
            print(f"match    - review page registered with PFMS: {url}")
        else:
            print(f"match    - could not register review page with PFMS: {err}")

    def submit_review(
        self, match_id: str, alliance: str, score: int, auto_score: int, reviewer: str
    ) -> tuple[bool, str]:
        """Report a human-reviewed final score to PFMS. Returns (ok, error)."""
        return self._post(
            "/api/match-review",
            {
                "matchId": match_id,
                "alliance": alliance,
                "score": score,
                "autoScore": auto_score,
                "reviewer": reviewer,
            },
            attempts=2,
        )
