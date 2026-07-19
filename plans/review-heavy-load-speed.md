# Review auto-speed: extra slowdown under heavy ball load

## Goal
Auto-speed playback in both reviewers (clips page + match review page) currently
bottoms out at ¼× during motion. When many balls flow at once that's still too
fast to count. Add a fourth zone — "heavy load" — that plays at ⅛× ("half
again" slower) when the motion signal spans multiple balls' worth of area.

## How "heavy load" is detected
- The per-frame `signal` in clip/match sidecars is moving yellow pixels in the
  goal zone (`counter.py`). `ball_area` (goal config) is one ball's pixel area;
  the counter estimates `n_balls = round(area / ball_area)`.
- Heavy threshold: `signal >= 1.5 * ball_area` — the same rounding boundary
  where the counter would call it 2+ balls.
- Sidecars don't store `ball_area`, so the API attaches it at serve time from
  the registered goal's config (`state.find_goal(name).config.ball_area`).
  Serve-time lookup means old clips benefit from corrected config. If the goal
  isn't registered (shouldn't happen in practice), the field is absent and the
  client behaves exactly as before (no heavy zone).

## Changes (all in src/ball_counter/web.py)
- `api_clip_detail`: add `d["ball_area"]` from goal config.
- `api_match_detail`: add `info["ball_area"]` per goal.
- Clips reviewer JS: `buildSpeedMap(signal, fps, ballArea)` marks zone 3 where
  signal ≥ 1.5×ball_area; poller maps 3→0.125. SPEEDS gains 0.125; ⅛× manual
  button added.
- Match reviewer JS: same three changes (it has its own copy of the logic);
  manual speed buttons get fraction labels (⅛ ¼ ½).

## Decisions
- ⅛× (0.125) chosen for "half again" of the ¼× floor. Browsers support
  playbackRate down to 0.0625, so no compat issue.
- Manual SPEEDS also gain 0.125 so `,`/`.` stepping doesn't jump to an extreme
  when auto-speed left the rate at 0.125.

## Findings / gotchas
- `scripts/check_syntax.py` fails on web.py under Windows (cp1252 default
  encoding vs the UTF-8 ¼×/½× glyphs). Use `python -m py_compile` instead.
- The embedded JS has TWO independent copies of the auto-speed logic (clips
  reviewer ~line 2055, match reviewer ~line 3140) — change both.
- New `scripts/check_embedded_js.py` extracts `<script>` blocks and runs
  `node --check` (stubs f-string interpolations). New
  `scripts/test_review_speed.py` (run with `uv run --with httpx python …`)
  exercises the real buildSpeedMap in node + the API payloads via TestClient.

## Progress
- [x] Locate both auto-speed implementations and the signal/ball_area sources
- [x] Server: expose ball_area in clip + match payloads
- [x] Client: zone 3 in both reviewers (+ ⅛× manual button)
- [x] Syntax check (py_compile + node --check) and functional test — all pass
- [x] Committed
