"""Draw a polygon ROI on a video frame interactively.

Click points to define the polygon outline. Press Enter to confirm, R to reset,
Backspace to undo last point, Q to quit.
Saves the polygon coordinates to a JSON file.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Draw a polygon ROI on a video frame")
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("--frame", type=int, default=None, help="Frame number to use (default: mid-clip)")
    parser.add_argument("--output", help="Output JSON file (default: <video>.roi.json)")
    args = parser.parse_args()

    video_path = Path(args.video)
    output_path = Path(args.output) if args.output else video_path.with_suffix(".roi.json")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: cannot open {video_path}", file=sys.stderr)
        sys.exit(1)

    if args.frame is not None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    else:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)

    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Error: cannot read frame", file=sys.stderr)
        sys.exit(1)

    points: list[tuple[int, int]] = []
    window = "Draw ROI - click points, Enter=save, Backspace=undo, R=reset, Q=quit"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, frame.shape[1] * 3, frame.shape[0] * 3)

    def redraw():
        display = frame.copy()
        if len(points) >= 2:
            # Draw filled polygon semi-transparent
            overlay = display.copy()
            pts = np.array(points, dtype=np.int32)
            cv2.fillPoly(overlay, [pts], (0, 180, 0))
            cv2.addWeighted(overlay, 0.3, display, 0.7, 0, display)
            # Draw outline
            cv2.polylines(display, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        elif len(points) == 1:
            pass  # just the dot below

        for i, pt in enumerate(points):
            cv2.circle(display, pt, 5, (0, 0, 255), -1)
            cv2.putText(display, str(i + 1), (pt[0] + 8, pt[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Instructions
        cv2.rectangle(display, (0, 0), (display.shape[1], 30), (0, 0, 0), -1)
        cv2.putText(display, f"{len(points)} points | Enter=save  Backspace=undo  R=reset  Q=quit",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow(window, display)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            redraw()

    cv2.setMouseCallback(window, on_mouse)
    redraw()

    print("Click points to outline the goal opening.")
    print("Enter=save, Backspace=undo last point, R=reset all, Q=quit")

    while True:
        key = cv2.waitKey(50) & 0xFF
        if key == ord("q"):
            cv2.destroyAllWindows()
            return
        elif key == ord("r"):
            points.clear()
            redraw()
        elif key == 8 and points:  # backspace
            points.pop()
            redraw()
        elif key == 13 and len(points) >= 3:
            break

    cv2.destroyAllWindows()

    data = {
        "roi": [list(pt) for pt in points],
    }
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved ROI ({len(points)} points) to {output_path}")


if __name__ == "__main__":
    main()
