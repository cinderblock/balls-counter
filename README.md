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

Install and enable the service:

```bash
sudo cp balls-counter.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now balls-counter
```

Edit [balls-counter.env](balls-counter.env) to change settings (config path, web port, YOLO model), then restart:

```bash
sudo systemctl restart balls-counter
```

View logs:

```bash
journalctl -u balls-counter -f
```

## Architecture

```
src/ball_counter/
  detector.py   - HSV color masking (create_mask, detect_balls)
  counter.py    - MotionCounter (line band + ROI ring modes)
  config.py     - Per-stream JSON configuration
  stream.py     - StreamProcessor (wraps MotionCounter + video capture)
  main.py       - Multi-stream runner with tiled display
  calibrate.py  - Interactive HSV calibration
```
