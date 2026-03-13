"""Draw a counting line on a video frame interactively.

Click two points to define the line. Press Enter to confirm, R to reset, Q to quit.
Saves the line coordinates to a JSON file.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2


def main():
    parser = argparse.ArgumentParser(description="Draw a counting line on a video frame")
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("--frame", type=int, default=None, help="Frame number to use (default: mid-clip)")
    parser.add_argument("--output", help="Output JSON file (default: <video>.line.json)")
    args = parser.parse_args()

    video_path = Path(args.video)
    output_path = Path(args.output) if args.output else video_path.with_suffix(".line.json")

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
    window = "Draw counting line - click 2 points, Enter=save, R=reset, Q=quit"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    def redraw():
        display = frame.copy()
        for pt in points:
            cv2.circle(display, pt, 6, (0, 0, 255), -1)
        if len(points) == 2:
            cv2.line(display, points[0], points[1], (0, 0, 255), 2)
            # Draw direction arrow (normal to line, indicating "scored" direction)
            mx = (points[0][0] + points[1][0]) // 2
            my = (points[0][1] + points[1][1]) // 2
            dx = points[1][0] - points[0][0]
            dy = points[1][1] - points[0][1]
            # Normal vector (right-hand side = "in" direction)
            nx, ny = -dy, dx
            length = (nx * nx + ny * ny) ** 0.5
            if length > 0:
                nx, ny = int(nx / length * 30), int(ny / length * 30)
                cv2.arrowedLine(display, (mx, my), (mx + nx, my + ny), (0, 255, 0), 2)
                cv2.putText(display, "IN", (mx + nx + 5, my + ny + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow(window, display)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
            points.append((x, y))
            redraw()

    cv2.setMouseCallback(window, on_mouse)
    redraw()

    print("Click two points to define the counting line.")
    print("The green arrow shows the 'IN' (scored) direction.")
    print("Enter=save, R=reset, Q=quit")

    while True:
        key = cv2.waitKey(50) & 0xFF
        if key == ord("q"):
            cv2.destroyAllWindows()
            return
        elif key == ord("r"):
            points.clear()
            redraw()
        elif key == 13 and len(points) == 2:
            break

    cv2.destroyAllWindows()

    data = {
        "line": [list(points[0]), list(points[1])],
    }
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved line to {output_path}: {points[0]} -> {points[1]}")


if __name__ == "__main__":
    main()
