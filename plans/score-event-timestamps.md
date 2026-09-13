# Score events carry the time the ball actually scored (counter side)

The full plan, decisions, and progress log live in the pFMS repo:
`~/git/practice-field-configurator/plans/score-event-timestamps.md`.

Counter-side summary (implemented 2026-09-13):

- `MotionEvent` gains `lag_frames` (how many frames before the detector
  fired the ball actually crossed: the signal peak for the threshold
  detector, the frame the track first appeared in the goal for YOLO, the
  peak for the ML detector) plus `mono` / `wall`, the read time of that
  frame, filled in by `GoalProcessor`.
- `GoalProcessor` keeps a ring of per-frame read times (monotonic and
  wall clock, stamped once per frame in `StreamProcessor.process_frame`)
  so an event can be dated `lag_frames` back.
- `PfmsForwarder` is a queued worker: a burst goes out as one POST,
  failures are retried with backoff (4xx is not retried), and `ageMs` is
  computed at each send attempt from the event's monotonic time.
  `pfms_capture_latency_ms` (config, default 0) is added to every age to
  cover camera → decoder latency. The worker never dies on an exception,
  and its console output is ASCII (a `→` killed it under cp1252).
- `tally_marks` gives a 3 s grace after an operator pause begins (ball
  in flight counts toward the interrupted period), matching pFMS. A goal
  that was already off (shift) gets no fresh grace from a pause.
- Smoke tests: `uv run python scripts/smoke_pfms_forwarder.py` (fake
  pFMS on localhost) and `uv run --extra web python scripts/smoke_match.py`.

Open: measure `pfms_capture_latency_ms` on sentinel and set it in
`config.json`.
