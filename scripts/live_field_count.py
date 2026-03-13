"""Live field ball counter: shows RTSP stream with zone overlays and ball counts.

Refreshes ball count every N frames. Press Q to quit, S to save snapshot.
"""

import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, "src")
from ball_counter.detector import create_mask

URL = "rtsp://10.255.9.97:8554/the-field"
ZONES_PATH = "samples/field_zones.json"
COUNT_INTERVAL = 30  # recount every N frames


def load_zones():
    if os.path.exists(ZONES_PATH):
        with open(ZONES_PATH) as f:
            return json.load(f)["zones"]
    return {
        "red": [0, 1350],
        "middle": [1450, 3050],
        "blue": [3150, 4536],
    }


def count_blobs(yellow_mask, x_start, x_end, min_area=100):
    zone = yellow_mask[:, x_start:x_end]
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(zone, connectivity=8)
    return sum(1 for i in range(1, n_labels) if stats[i, cv2.CC_STAT_AREA] >= min_area)


def main():
    zones = load_zones()
    zone_colors = {
        "red": (0, 0, 180),
        "middle": (180, 180, 0),
        "blue": (180, 0, 0),
    }

    print(f"Connecting to {URL}...")
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
    cap = cv2.VideoCapture(URL, cv2.CAP_FFMPEG)

    if not cap.isOpened():
        print("Error: cannot open stream", file=sys.stderr)
        sys.exit(1)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Stream: {w}x{h}")

    window = "Field Ball Counter - Q=quit S=snapshot"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, min(w, 1920), min(h, 1080))

    frame_idx = 0
    counts = {name: 0 for name in zones}

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Stream lost")
            break

        # Recount periodically
        if frame_idx % COUNT_INTERVAL == 0:
            yellow = create_mask(frame)
            for name, (x_start, x_end) in zones.items():
                counts[name] = count_blobs(yellow, x_start, x_end)

        # Draw overlay
        display = frame.copy()
        total = 0
        for name, (x_start, x_end) in zones.items():
            color = zone_colors.get(name, (128, 128, 128))
            # Zone tint
            overlay = display.copy()
            cv2.rectangle(overlay, (x_start, 0), (x_end, h), color, -1)
            cv2.addWeighted(overlay, 0.1, display, 0.9, 0, display)
            # Zone boundary
            cv2.line(display, (x_start, 0), (x_start, h), (0, 255, 255), 2)
            cv2.line(display, (x_end, 0), (x_end, h), (0, 255, 255), 2)
            # Count label
            cx = (x_start + x_end) // 2
            cv2.putText(display, f"{name.upper()}: {counts[name]}",
                        (cx - 100, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 2.0,
                        (255, 255, 255), 4)
            total += counts[name]

        # Header
        cv2.rectangle(display, (0, 0), (w, 50), (0, 0, 0), -1)
        cv2.putText(display, f"Total: {total}  |  Red: {counts.get('red', 0)}  "
                    f"Mid: {counts.get('middle', 0)}  Blue: {counts.get('blue', 0)}",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

        cv2.imshow(window, display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            path = f"samples/field_live_{frame_idx}.png"
            cv2.imwrite(path, display)
            print(f"Saved {path}")

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
