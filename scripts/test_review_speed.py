#!/usr/bin/env python
"""Test the review auto-speed heavy-load zone end to end (no cameras needed).

1. Extracts the real buildSpeedMap + rate mapping from web.py's embedded JS and
   runs them in node against a synthetic signal (quiet / one ball / heavy flow),
   asserting the zone map and playback rates for both reviewer pages.
2. Spins up the FastAPI app with a fake goal registered and asserts the clip and
   match detail endpoints attach ball_area from goal config.

Run with: uv run --with httpx python scripts/test_review_speed.py
"""
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

WEB_PY = Path(__file__).parent.parent / "src" / "ball_counter" / "web.py"
BALL_AREA = 900
FPS = 30


def check_js():
    text = WEB_PY.read_text(encoding="utf-8")
    fns = re.findall(r"function buildSpeedMap\(.*?return buf;\s*\n\s*\}", text, re.DOTALL)
    assert len(fns) == 2, f"expected 2 buildSpeedMap copies, found {len(fns)}"

    # signal: 2s quiet, 1s one-ball motion, 2s quiet, 1s heavy (2+ balls), 10s quiet
    signal = (
        [0] * (2 * FPS)
        + [BALL_AREA] * FPS
        + [0] * (2 * FPS)
        + [BALL_AREA * 2] * FPS
        + [0] * (10 * FPS)
    )
    for i, fn in enumerate(fns):
        harness = f"""
{fn}
const signal = {json.dumps(signal)};
const map = buildSpeedMap(signal, {FPS}, {BALL_AREA});
const fps = {FPS};
const at = sec => map[Math.round(sec * fps)];
const assert = (cond, msg) => {{ if (!cond) {{ console.error('FAIL: ' + msg); process.exit(1); }} }};
assert(at(2.5) === 2, 'one-ball motion should be zone 2, got ' + at(2.5));
assert(at(5.5) === 3, 'heavy flow should be zone 3, got ' + at(5.5));
assert(at(1.0) === 1, 'near-motion should be zone 1, got ' + at(1.0));
assert(at(15.5) === 0, 'quiet tail should be zone 0, got ' + at(15.5));
// Without ballArea (old server payload), heavy zone must not appear
const map2 = buildSpeedMap(signal, fps, undefined);
assert(map2[Math.round(5.5 * fps)] === 2, 'no ballArea -> heavy stays zone 2');
console.log('buildSpeedMap copy OK');
"""
        with tempfile.NamedTemporaryFile(
            "w", suffix=".js", delete=False, encoding="utf-8"
        ) as f:
            f.write(harness)
            tmp = Path(f.name)
        try:
            r = subprocess.run(["node", str(tmp)], capture_output=True, text=True, shell=True)
            if r.returncode != 0:
                print(f"buildSpeedMap copy {i + 1}: {r.stderr or r.stdout}", file=sys.stderr)
                sys.exit(1)
        finally:
            tmp.unlink(missing_ok=True)

    # Both pollers must map zone 3 -> 0.125x
    assert text.count("0.125") >= 4, "0.125x rate mapping missing"
    print("JS: zone map and heavy-load threshold OK (both reviewer copies)")


def check_api():
    from starlette.testclient import TestClient
    from ball_counter.web import AppState, create_app

    state = AppState()
    fake_goal = SimpleNamespace(config=SimpleNamespace(ball_area=BALL_AREA))
    state.register_goal("red-goal", fake_goal)

    frames = [
        {"frame_idx": i, "timestamp": "t", "signal": s, "rising": False, "event": None}
        for i, s in enumerate([0, BALL_AREA * 2, 0])
    ]

    with tempfile.TemporaryDirectory() as td:
        clips_dir = Path(td)
        (clips_dir / "clip1.json").write_text(
            json.dumps({"goal": "red-goal", "fps": FPS, "n_frames": 3, "frames": frames})
        )
        state.set_clips_dir(clips_dir)

        matches_dir = clips_dir / "matches"
        matches_dir.mkdir()
        (matches_dir / "m1.match.json").write_text(json.dumps({
            "match_id": "m1",
            "goals": {"red-goal": {"clip": "m1_red-goal", "fps": FPS, "n_frames": 3}},
            "timeline": [],
        }))
        (matches_dir / "m1_red-goal.json").write_text(json.dumps({"frames": frames}))
        state.set_match_context(matches_dir, None, None)

        client = TestClient(create_app(state))

        d = client.get("/api/clips/clip1").json()
        assert d["ball_area"] == BALL_AREA, f"clip ball_area: {d.get('ball_area')}"
        assert d["signal"] == [0, BALL_AREA * 2, 0]

        m = client.get("/api/match/m1").json()
        g = m["goals"]["red-goal"]
        assert g["ball_area"] == BALL_AREA, f"match ball_area: {g.get('ball_area')}"
        assert g["signal"] == [0, BALL_AREA * 2, 0]

        # Unregistered goal -> field absent, client falls back to old behavior
        (clips_dir / "clip2.json").write_text(
            json.dumps({"goal": "unknown-goal", "fps": FPS, "n_frames": 3, "frames": frames})
        )
        d2 = client.get("/api/clips/clip2").json()
        assert "ball_area" not in d2, "unregistered goal should omit ball_area"

    print("API: ball_area attached to clip + match payloads OK")


if __name__ == "__main__":
    check_js()
    check_api()
    print("all checks passed")
