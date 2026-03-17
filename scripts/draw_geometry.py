"""Draw lines or polygons on a still frame from a video.

Usage:
  uv run python scripts/draw_geometry.py <video> [--frame N] [--crop x1,y1,x2,y2]

Controls:
  L        - start drawing a LINE (click 2 points)
  P        - start drawing a POLYGON (click points, Enter to close)
  Backspace - undo last point
  R        - remove last geometry
  Enter    - finish current shape / save and quit
  Q        - quit without saving
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Draw geometry on a video frame")
    parser.add_argument("video", help="Video file path")
    parser.add_argument("--frame", type=int, default=50, help="Frame number to show")
    parser.add_argument("--crop", help="Crop region: x1,y1,x2,y2")
    parser.add_argument("--output", help="Output JSON path (default: <video>.geometry.json)")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Cannot open {args.video}", file=sys.stderr)
        sys.exit(1)

    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print(f"Cannot read frame {args.frame}", file=sys.stderr)
        sys.exit(1)

    crop = None
    if args.crop:
        x1, y1, x2, y2 = map(int, args.crop.split(","))
        crop = (x1, y1, x2, y2)
        frame = frame[y1:y2, x1:x2]

    out_path = args.output or str(Path(args.video).with_suffix(".geometry.json"))

    # Load existing geometries
    geometries = []
    if Path(out_path).exists():
        with open(out_path) as f:
            geometries = json.load(f)
        print(f"Loaded {len(geometries)} existing geometries from {out_path}")

    h, w = frame.shape[:2]
    current_points = []
    current_mode = None  # "line" or "polygon"
    line_count = sum(1 for g in geometries if g["type"] == "line")
    poly_count = sum(1 for g in geometries if g["type"] == "polygon")

    window = "Draw geometry - L=line P=polygon Enter=save Q=quit"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    scale = min(1920 / w, 1080 / h, 3.0)
    cv2.resizeWindow(window, int(w * scale), int(h * scale))

    def to_full(pt):
        """Convert crop-relative point back to full-frame coords."""
        if crop:
            return [pt[0] + crop[0], pt[1] + crop[1]]
        return list(pt)

    def to_crop(pt):
        """Convert full-frame point to crop-relative coords."""
        if crop:
            return [pt[0] - crop[0], pt[1] - crop[1]]
        return list(pt)

    def redraw():
        display = frame.copy()

        # Draw existing geometries
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (0, 255, 255)]
        for i, geom in enumerate(geometries):
            color = colors[i % len(colors)]
            pts = [to_crop(p) for p in geom["points"]]
            if geom["type"] == "line":
                cv2.line(display, tuple(pts[0]), tuple(pts[1]), color, 2)
                for pt in pts:
                    cv2.circle(display, tuple(pt), 5, color, -1)
            elif geom["type"] == "polygon":
                arr = np.array(pts, dtype=np.int32)
                overlay = display.copy()
                cv2.fillPoly(overlay, [arr], color)
                cv2.addWeighted(overlay, 0.2, display, 0.8, 0, display)
                cv2.polylines(display, [arr], True, color, 2)
            cv2.putText(display, geom["name"], (pts[0][0] + 8, pts[0][1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Draw current points
        if current_points:
            for pt in current_points:
                cv2.circle(display, tuple(pt), 5, (255, 0, 255), -1)
            if len(current_points) >= 2:
                if current_mode == "line":
                    cv2.line(display, tuple(current_points[0]), tuple(current_points[1]), (255, 0, 255), 2)
                else:
                    cv2.polylines(display, [np.array(current_points, dtype=np.int32)], False, (255, 0, 255), 2)

        # Status bar
        cv2.rectangle(display, (0, 0), (w, 30), (0, 0, 0), -1)
        mode_str = f"Drawing {current_mode.upper()}" if current_mode else "L=line P=polygon"
        status = f"{len(geometries)} shapes | {mode_str} | {len(current_points)} pts"
        cv2.putText(display, status, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow(window, display)

    def on_mouse(event, x, y, flags, param):
        nonlocal current_points
        if event != cv2.EVENT_LBUTTONDOWN or current_mode is None:
            return
        if current_mode == "line" and len(current_points) >= 2:
            return
        current_points.append([x, y])
        redraw()

    cv2.setMouseCallback(window, on_mouse)
    redraw()

    while True:
        key = cv2.waitKey(50) & 0xFF

        if key == ord("q"):
            print("Quit without saving")
            break

        elif key == ord("l") and current_mode is None:
            current_mode = "line"
            current_points = []
            redraw()

        elif key == ord("p") and current_mode is None:
            current_mode = "polygon"
            current_points = []
            redraw()

        elif key == 8:  # Backspace
            if current_points:
                current_points.pop()
                redraw()
            elif geometries:
                removed = geometries.pop()
                print(f"Removed {removed['name']}")
                if removed["type"] == "line":
                    line_count -= 1
                else:
                    poly_count -= 1
                redraw()

        elif key == ord("r") and current_mode is None and geometries:
            removed = geometries.pop()
            print(f"Removed {removed['name']}")
            if removed["type"] == "line":
                line_count -= 1
            else:
                poly_count -= 1
            redraw()

        elif key == 13:  # Enter
            if current_mode == "line" and len(current_points) == 2:
                line_count += 1
                name = f"line-{line_count}"
                geometries.append({
                    "type": "line",
                    "points": [to_full(p) for p in current_points],
                    "name": name,
                })
                print(f"Added {name}")
                current_mode = None
                current_points = []
                redraw()

            elif current_mode == "polygon" and len(current_points) >= 3:
                poly_count += 1
                name = f"polygon-{poly_count}"
                geometries.append({
                    "type": "polygon",
                    "points": [to_full(p) for p in current_points],
                    "name": name,
                })
                print(f"Added {name}")
                current_mode = None
                current_points = []
                redraw()

            elif current_mode is None:
                # Save and quit
                with open(out_path, "w") as f:
                    json.dump(geometries, f, indent=2)
                print(f"Saved {len(geometries)} geometries to {out_path}")
                break

        elif key == 27:  # Escape - cancel current shape
            current_mode = None
            current_points = []
            redraw()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
