# Ball Counter

Computer vision system to count ball scoring events in FIRST Robotics competitions. Detects bright yellow spheres using motion-based background subtraction combined with HSV color thresholding.

Supports multiple simultaneous camera streams with independent tuning. Two geometry modes:

- **Line band**: a narrow band around a counting line — for side-view cameras watching balls drop through an outlet
- **ROI ring**: a thin ring around a polygon perimeter — for top-down cameras watching balls enter a goal opening

Both use the same core algorithm: MOG2 background subtraction isolates moving pixels, HSV masking isolates yellow, and peak detection on the "moving yellow in zone" signal triggers scoring events.

## Setup

```bash
uv sync
```

## Usage

### 1. Draw counting geometry

For side-view cameras (outlet), draw a counting line:
```bash
uv run python scripts/draw_line.py path/to/video.mp4
```

For top-down cameras (inlet), draw the goal opening polygon:
```bash
uv run python scripts/draw_roi.py path/to/video.mp4
```

### 2. Create a config file

Copy `config.example.json` to `config.json`. Each stream needs:
- `source`: RTSP URL or video file path
- `line` or `roi_points`: counting geometry from step 1
- `ball_area`, `band_width`, `cooldown`: tuning parameters

### 3. Run the counter

```bash
uv run ball-counter config.json
```

Options:

| Flag | Description |
|------|-------------|
| `--web-port PORT_OR_SOCKET` | Enable web UI on a TCP port (e.g. `8080`) or Unix socket path |
| `--host HOST` | Interface to bind the web server to (default: `0.0.0.0`) |
| `--trusted-proxies IPS` | Comma-separated IPs (or `*`) to trust for `X-Forwarded-*` headers |
| `--yolo-model PATH` | YOLO ball detector model — uses object tracking instead of signal peak detection |
| `--model PATH` | Trained ML peak detector — replaces threshold-based counting |
| `--wizard` | Launch setup wizard even if config already exists |
| `--progress-interval N` | Print video-file progress every N frames (default: 300, 0 = off) |

All streams are viewable via the web UI with real-time signal overlay and running counts.

### 4. Calibrate HSV thresholds (optional)

```bash
uv run python -m ball_counter.calibrate path/to/video.mp4
```

### Match recording & review (pFMS)

When the config has a `pfms_url`, the counter follows the pFMS match state
over its public WebSocket (`/ws/scores`) and records **every goal for the
whole match** — countdown pre-roll, auto, pauses, teleop, endgame, and the
post-match ball-count period. The phase timeline (including operator
pauses) is stored with each recording, so reviewers can see exactly which
video spans count toward the score.

- `/matches` lists match recordings (a running match shows as LIVE)
- `/match/<id>` is the per-match review page: pick the red or blue goal
  (two reviewers can work in parallel), watch the cropped goal video with
  the match periods shaded on the timeline, and mark each scored ball —
  live during the match or after the fact. Marks during countdown or an
  operator pause are automatically excluded from the tally.
- **Submit final score to PFMS** reports the reviewed score (with the auto
  portion split out) back to pFMS match history via `POST /api/match-review`
  using the same `pfms_url`/`pfms_key` as live score forwarding.

Config keys:

```jsonc
{
  "pfms_url": "http://pfms.tsl",       // pFMS base URL
  "pfms_key": "…",                     // API key (optional)
  "public_url": "http://sentinel:8080" // this service's own URL — lets pFMS
                                       // link its match history to /match/<id>
}
```

Recordings are saved under `<config dir>/clips/matches/` as one MP4 + JSON
sidecar per goal plus a `<match id>.match.json` metadata file.

### Field zone counter

Count total balls in 3 field zones (red/middle/blue) from a stitched overhead RTSP stream:

```bash
uv run python scripts/count_field_zones.py
uv run python scripts/live_field_count.py  # live view
```

## Tools

| Script | Purpose |
|--------|---------|
| `scripts/draw_line.py` | Draw a counting line on a video frame |
| `scripts/draw_roi.py` | Draw a polygon ROI on a video frame |
| `scripts/draw_zones.py` | Draw field zone boundaries interactively |
| `scripts/annotate.py` | Frame-by-frame score annotation for ground truth |
| `scripts/count_field_zones.py` | Snapshot ball count by field zone |
| `scripts/live_field_count.py` | Live RTSP field zone counter |

## Running as a systemd service

Runs as a **user** service (no root needed). Requires `ffmpeg` on the host for
GPU (NVDEC) decode — install it with your package manager (e.g. `sudo apt install ffmpeg`).

Install and enable the service:

```bash
mkdir -p ~/.config/systemd/user
cp balls-counter.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now balls-counter
# run at boot without an active login session:
sudo loginctl enable-linger "$USER"
```

Edit [balls-counter.env](balls-counter.env) to change settings (config path, web port, YOLO model), then restart:

```bash
systemctl --user restart balls-counter
```

View logs:

```bash
journalctl --user -u balls-counter -f
```

## Architecture

```
src/ball_counter/
  detector.py   - HSV color masking (create_mask, detect_balls)
  counter.py    - MotionCounter (line band + ROI ring modes)
  config.py     - Per-stream JSON configuration
  stream.py     - StreamProcessor (wraps MotionCounter + video capture)
  buffer.py     - Rolling per-goal frame buffer (feeds clips + recordings)
  clips.py      - Clip saving/trimming (MP4 + JSON sidecar)
  match.py      - pFMS match-scoped recording + review score report-back
  pfms.py       - Live score forwarding to pFMS
  web.py        - HTTP API + web UIs (live, clips review, match review, wizard)
  main.py       - Multi-stream runner with tiled display
  calibrate.py  - Interactive HSV calibration
```
