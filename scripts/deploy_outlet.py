"""Deploy the motion-based outlet counter on a live RTSP stream.

Usage:
  uv run python scripts/deploy_outlet.py rtsp://10.255.9.97:8554/red-goal
  uv run python scripts/deploy_outlet.py rtsp://... --line-config red-outlet.line.json
  uv run python scripts/deploy_outlet.py rtsp://... --setup  # draw line interactively

First run: use --setup to draw the counting line on a live frame.
Subsequent runs: reuse the saved line config.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, "src")
from ball_counter.counter import MotionCounter


def grab_clean_frame(url):
    """Grab a clean frame from RTSP, waiting for decoder to stabilize."""
    print(f"Connecting to {url}...")
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)

    if not cap.isOpened():
        print("Error: cannot open stream", file=sys.stderr)
        sys.exit(1)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Stream: {w}x{h} @ {fps}fps")
    print("Waiting for clean frame...")

    for i in range(150):
        ret, frame = cap.read()
        if not ret:
            print(f"Lost stream at frame {i}", file=sys.stderr)
            sys.exit(1)

    return cap, frame, fps


def draw_line_interactive(frame):
    """Let the user draw a counting line on a frame."""
    points = []
    window = "Draw counting line - click 2 points, Enter=confirm, R=reset"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    h, w = frame.shape[:2]
    cv2.resizeWindow(window, min(w * 2, 1920), min(h * 2, 1080))

    def redraw():
        display = frame.copy()
        for pt in points:
            cv2.circle(display, tuple(pt), 6, (0, 0, 255), -1)
        if len(points) == 2:
            cv2.line(display, tuple(points[0]), tuple(points[1]), (0, 0, 255), 2)
            # Band preview
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

    print("Click two points to define the counting line.")
    print("Enter=confirm, R=reset, Q=cancel")

    while True:
        key = cv2.waitKey(50) & 0xFF
        if key == ord("q"):
            cv2.destroyWindow(window)
            sys.exit(0)
        elif key == ord("r"):
            points.clear()
            redraw()
        elif key == 13 and len(points) == 2:
            break

    cv2.destroyWindow(window)
    return points


def run_counter(cap, fps, line, ball_area, band_width, min_peak, fall_ratio):
    """Run the live counter with visualization."""
    ret, frame = cap.read()
    if not ret:
        print("Cannot read frame")
        return
    h, w = frame.shape[:2]

    counter = MotionCounter(
        frame_shape=(h, w),
        line=tuple(line),
        ball_area=ball_area,
        band_width=band_width,
        min_peak=min_peak,
        fall_ratio=fall_ratio,
    )

    window = "Ball Counter - Q=quit"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, min(w * 3, 1920), min(h * 3, 1080))

    # Process the frame we already read
    counter.process_frame(frame)

    blips = []
    BLIP_DURATION = 20

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Stream lost, reconnecting not implemented yet")
            break

        event = counter.process_frame(frame)
        if event:
            blips.append([event.n_balls, BLIP_DURATION])
            print(f"  +{event.n_balls} (peak={event.peak_area}) Total: {counter.count}")

        # Draw overlay
        display = frame.copy()

        # Draw counting line and band
        counter.draw(display)

        # Score blips
        active_blips = []
        for blip in blips:
            n, remaining = blip
            if remaining > 0:
                alpha = remaining / BLIP_DURATION
                overlay = display.copy()
                p1, p2 = line
                cv2.line(overlay, tuple(p1), tuple(p2), (0, 255, 0), 4)
                cv2.addWeighted(overlay, alpha * 0.5, display, 1.0 - alpha * 0.5, 0, display)
                blip[1] -= 1
                active_blips.append(blip)
        blips = active_blips

        # Header
        cv2.rectangle(display, (0, 0), (w, 60), (0, 0, 0), -1)
        cv2.putText(display, f"Count: {counter.count}",
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

    print(f"\nFinal count: {counter.count}")
    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Deploy outlet ball counter on live RTSP stream")
    parser.add_argument("url", help="RTSP stream URL")
    parser.add_argument("--setup", action="store_true",
                        help="Draw counting line interactively")
    parser.add_argument("--line-config", default=None,
                        help="Path to line config JSON (default: <url-slug>.line.json)")
    parser.add_argument("--ball-area", type=int, default=900)
    parser.add_argument("--band-width", type=int, default=20)
    parser.add_argument("--min-peak", type=int, default=0)
    parser.add_argument("--fall-ratio", type=float, default=0.5)
    args = parser.parse_args()

    # Determine config path
    if args.line_config:
        config_path = Path(args.line_config)
    else:
        slug = args.url.replace("://", "_").replace("/", "_").replace(":", "_")
        config_path = Path(f"configs/{slug}.line.json")

    # Connect and get a clean frame
    cap, frame, fps = grab_clean_frame(args.url)

    # Setup or load line
    if args.setup or not config_path.exists():
        if not args.setup and not config_path.exists():
            print(f"No line config at {config_path}, entering setup mode...")
        line = draw_line_interactive(frame)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w") as f:
            json.dump({"line": line}, f, indent=2)
        print(f"Saved line to {config_path}")
    else:
        with open(config_path) as f:
            data = json.load(f)
        line = data["line"]
        print(f"Loaded line from {config_path}: {line}")

    print(f"\nStarting counter (ball_area={args.ball_area}, band_width={args.band_width})...")
    run_counter(cap, fps, line, args.ball_area, args.band_width,
                args.min_peak, args.fall_ratio)


if __name__ == "__main__":
    main()
