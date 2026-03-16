"""Deploy ball counter on a live RTSP stream.

Usage:
  # First run — interactive setup
  uv run python scripts/deploy.py rtsp://... --name red-outlet --setup line
  uv run python scripts/deploy.py rtsp://... --name blue-inlet --setup roi

  # Subsequent runs — reuse saved config
  uv run python scripts/deploy.py rtsp://... --name red-outlet

  # Multi-stream from config file
  uv run python scripts/deploy.py --config config.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, "src")
from ball_counter.config import StreamConfig, load_configs
from ball_counter.counter import MotionCounter


CONFIGS_DIR = Path("configs")
BLIP_DURATION = 20


def open_stream(url):
    """Open RTSP stream with TCP transport, wait for clean frames."""
    print(f"Connecting to {url}...")
    if url.startswith("rtsp://"):
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    else:
        cap = cv2.VideoCapture(url)

    if not cap.isOpened():
        print(f"Error: cannot open {url}", file=sys.stderr)
        sys.exit(1)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Stream: {w}x{h} @ {fps}fps")

    if url.startswith("rtsp://"):
        print("Waiting for decoder to stabilize...")
        for i in range(150):
            ret, frame = cap.read()
            if not ret:
                print(f"Lost stream at frame {i}", file=sys.stderr)
                sys.exit(1)
    else:
        ret, frame = cap.read()
        if not ret:
            print("Cannot read first frame", file=sys.stderr)
            sys.exit(1)

    return cap, frame, fps


def setup_line(frame):
    """Interactive line drawing. Returns [[x1,y1],[x2,y2]]."""
    points = []
    h, w = frame.shape[:2]
    window = "Draw counting line - click 2 points, Enter=confirm, R=reset, Q=cancel"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, min(w * 2, 1920), min(h * 2, 1080))

    def redraw():
        display = frame.copy()
        for pt in points:
            cv2.circle(display, tuple(pt), 6, (0, 0, 255), -1)
        if len(points) == 2:
            cv2.line(display, tuple(points[0]), tuple(points[1]), (0, 0, 255), 2)
            p1, p2 = points
            band_pts = np.array([
                [p1[0], p1[1] - 20], [p2[0], p2[1] - 20],
                [p2[0], p2[1] + 20], [p1[0], p1[1] + 20]
            ], dtype=np.int32)
            overlay = display.copy()
            cv2.fillPoly(overlay, [band_pts], (0, 0, 180))
            cv2.addWeighted(overlay, 0.3, display, 0.7, 0, display)
        cv2.imshow(window, display)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
            points.append([x, y])
            redraw()

    cv2.setMouseCallback(window, on_mouse)
    redraw()

    while True:
        key = cv2.waitKey(50) & 0xFF
        if key == ord("q"):
            cv2.destroyWindow(window)
            return None
        elif key == ord("r"):
            points.clear()
            redraw()
        elif key == 13 and len(points) == 2:
            break

    cv2.destroyWindow(window)
    return points


def setup_roi(frame):
    """Interactive polygon drawing. Returns list of [x,y] points."""
    points = []
    h, w = frame.shape[:2]
    window = "Draw ROI polygon - click points, Enter=confirm, Backspace=undo, R=reset, Q=cancel"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, min(w * 2, 1920), min(h * 2, 1080))

    def redraw():
        display = frame.copy()
        if len(points) >= 2:
            pts = np.array(points, dtype=np.int32)
            overlay = display.copy()
            cv2.fillPoly(overlay, [pts], (0, 180, 0))
            cv2.addWeighted(overlay, 0.3, display, 0.7, 0, display)
            cv2.polylines(display, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        for i, pt in enumerate(points):
            cv2.circle(display, tuple(pt), 5, (0, 0, 255), -1)
            cv2.putText(display, str(i + 1), (pt[0] + 8, pt[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.rectangle(display, (0, 0), (w, 30), (0, 0, 0), -1)
        cv2.putText(display, f"{len(points)} points | Enter=save  Backspace=undo  R=reset  Q=cancel",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow(window, display)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append([x, y])
            redraw()

    cv2.setMouseCallback(window, on_mouse)
    redraw()

    while True:
        key = cv2.waitKey(50) & 0xFF
        if key == ord("q"):
            cv2.destroyWindow(window)
            return None
        elif key == ord("r"):
            points.clear()
            redraw()
        elif key == 8 and points:
            points.pop()
            redraw()
        elif key == 13 and len(points) >= 3:
            break

    cv2.destroyWindow(window)
    return points


def load_stream_config(name):
    """Load a saved stream config by name."""
    path = CONFIGS_DIR / f"{name}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def save_stream_config(name, data):
    """Save a stream config by name."""
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    path = CONFIGS_DIR / f"{name}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved config to {path}")


def run_single_stream(cap, fps, config, frame_shape):
    """Run counter on a single stream with visualization."""
    h, w = frame_shape

    line = config.get("line")
    roi = config.get("roi_points")

    counter = MotionCounter(
        frame_shape=(h, w),
        line=tuple(line) if line else None,
        roi=roi if roi else None,
        ball_area=config.get("ball_area", 900),
        band_width=config.get("band_width", 20),
        min_peak=config.get("min_peak", 0),
        fall_ratio=config.get("fall_ratio", 0.5),
        cooldown=config.get("cooldown", 0),
        hsv_low=tuple(config.get("hsv_low", [20, 100, 100])),
        hsv_high=tuple(config.get("hsv_high", [35, 255, 255])),
    )

    name = config.get("name", "counter")
    window = f"{name} - Q=quit"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, min(w * 3, 1920), min(h * 3, 1080))

    blips = []

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Stream ended or lost")
            break

        event = counter.process_frame(frame)
        if event:
            blips.append([event.n_balls, BLIP_DURATION])
            print(f"  [{name}] +{event.n_balls} (peak={event.peak_area}) Total: {counter.count}")

        display = frame.copy()
        counter.draw(display)

        # Score blips — flash line/roi green
        active_blips = []
        for blip in blips:
            n, remaining = blip
            if remaining > 0:
                alpha = remaining / BLIP_DURATION
                overlay = display.copy()
                if line:
                    cv2.line(overlay, tuple(line[0]), tuple(line[1]), (0, 255, 0), 4)
                cv2.addWeighted(overlay, alpha * 0.5, display, 1.0 - alpha * 0.5, 0, display)
                blip[1] -= 1
                active_blips.append(blip)
        blips = active_blips

        # Header
        cv2.rectangle(display, (0, 0), (w, 60), (0, 0, 0), -1)
        cv2.putText(display, f"{name}  Count: {counter.count}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        signal_text = f"Signal: {counter.signal:5d}px"
        if counter.rising:
            signal_text += " RISING"
        cv2.putText(display, signal_text,
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 200), 1)

        cv2.imshow(window, display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    print(f"\n[{name}] Final count: {counter.count}")
    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Deploy ball counter on live stream")
    parser.add_argument("url", nargs="?", help="Stream URL or video file")
    parser.add_argument("--name", default="stream", help="Stream name (used for config file)")
    parser.add_argument("--setup", choices=["line", "roi"],
                        help="Interactive setup mode: draw a line or ROI polygon")
    parser.add_argument("--config", help="Use a multi-stream config JSON file instead")
    parser.add_argument("--ball-area", type=int, default=None)
    parser.add_argument("--band-width", type=int, default=None)
    parser.add_argument("--min-peak", type=int, default=None)
    parser.add_argument("--fall-ratio", type=float, default=None)
    parser.add_argument("--cooldown", type=int, default=None)
    args = parser.parse_args()

    # Multi-stream mode
    if args.config:
        configs = load_configs(Path(args.config))
        # For now, run first stream only (multi-stream via main.py)
        if not configs:
            print("No streams in config", file=sys.stderr)
            sys.exit(1)
        cfg = configs[0]
        cap, frame, fps = open_stream(cfg.source)
        config_dict = {
            "name": cfg.name,
            "line": cfg.line,
            "roi_points": cfg.roi_points,
            "ball_area": cfg.ball_area,
            "band_width": cfg.band_width,
            "min_peak": cfg.min_peak,
            "fall_ratio": cfg.fall_ratio,
            "cooldown": cfg.cooldown,
            "hsv_low": list(cfg.hsv_low),
            "hsv_high": list(cfg.hsv_high),
        }
        run_single_stream(cap, fps, config_dict, frame.shape[:2])
        return

    # Single stream mode
    if not args.url:
        parser.error("url is required unless --config is used")

    cap, frame, fps = open_stream(args.url)

    # Load or create config
    saved = load_stream_config(args.name)

    if args.setup or saved is None:
        if saved is None and not args.setup:
            print(f"No saved config for '{args.name}'. What type of geometry?")
            print("  Rerun with: --setup line  (for side-view outlet)")
            print("  Rerun with: --setup roi   (for top-down inlet)")
            cap.release()
            sys.exit(1)

        if args.setup == "line":
            geom = setup_line(frame)
            if geom is None:
                sys.exit(0)
            saved = {"name": args.name, "line": geom}
        elif args.setup == "roi":
            geom = setup_roi(frame)
            if geom is None:
                sys.exit(0)
            saved = {"name": args.name, "roi_points": geom}

        # Apply CLI overrides before saving
        if args.ball_area is not None:
            saved["ball_area"] = args.ball_area
        if args.band_width is not None:
            saved["band_width"] = args.band_width
        if args.min_peak is not None:
            saved["min_peak"] = args.min_peak
        if args.fall_ratio is not None:
            saved["fall_ratio"] = args.fall_ratio
        if args.cooldown is not None:
            saved["cooldown"] = args.cooldown

        save_stream_config(args.name, saved)

    # Apply CLI overrides for this run
    config = dict(saved)
    config["name"] = args.name
    if args.ball_area is not None:
        config["ball_area"] = args.ball_area
    if args.band_width is not None:
        config["band_width"] = args.band_width
    if args.min_peak is not None:
        config["min_peak"] = args.min_peak
    if args.fall_ratio is not None:
        config["fall_ratio"] = args.fall_ratio
    if args.cooldown is not None:
        config["cooldown"] = args.cooldown

    print(f"\nConfig: {json.dumps({k: v for k, v in config.items() if k != 'name'}, indent=2)}")
    print("Starting counter...\n")

    run_single_stream(cap, fps, config, frame.shape[:2])


if __name__ == "__main__":
    main()
