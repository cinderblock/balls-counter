"""Count yellow balls in 3 field zones from RTSP stream snapshot.

Grabs a frame, splits into Red/Middle/Blue zones, counts yellow blobs in each.
Avoids camera seam overlap areas.
"""

import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, "src")
from ball_counter.detector import create_mask


def grab_frame(url="rtsp://10.255.9.97:8554/the-field", warmup_seconds=15):
    """Grab a clean frame from the RTSP stream using ffmpeg."""
    import subprocess
    import tempfile

    out_path = "samples/field_latest.png"
    cmd = [
        "ffmpeg", "-rtsp_transport", "tcp",
        "-i", url,
        "-ss", str(warmup_seconds),
        "-frames:v", "1",
        "-update", "1",
        "-y", out_path
    ]
    subprocess.run(cmd, capture_output=True, timeout=30)
    frame = cv2.imread(out_path)
    return frame


def count_balls_in_region(yellow_mask, x_start, x_end, min_area=100):
    """Count yellow blobs in a vertical strip."""
    zone = yellow_mask[:, x_start:x_end]
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        zone, connectivity=8)
    count = 0
    for i in range(1, n_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            count += 1
    return count


def main():
    # Check for saved zones or use defaults
    zones_path = "samples/field_zones.json"
    if os.path.exists(zones_path):
        with open(zones_path) as f:
            zones = json.load(f)["zones"]
        print(f"Using saved zones from {zones_path}")
    else:
        # Default zone boundaries estimated from the 4536px wide panorama
        # Seams appear around x~1400 and x~3100
        # Place dividers to avoid seams
        zones = {
            "red": [0, 1350],
            "middle": [1450, 3050],
            "blue": [3150, 4536],
        }
        print("Using default zone boundaries (run draw_zones.py to customize)")

    print(f"Zones: {json.dumps(zones)}")

    # Grab frame
    print("\nGrabbing frame from stream...")
    frame = grab_frame()
    if frame is None:
        print("Error: could not grab frame", file=sys.stderr)
        sys.exit(1)

    h, w = frame.shape[:2]
    print(f"Frame: {w}x{h}")

    # Detect yellow
    yellow = create_mask(frame)

    # Count per zone
    print(f"\n{'='*40}")
    print(f"  FIELD BALL COUNT")
    print(f"{'='*40}")
    total = 0
    for name, (x_start, x_end) in zones.items():
        count = count_balls_in_region(yellow, x_start, x_end)
        total += count
        print(f"  {name:>6s}: {count:3d} balls  (x={x_start}-{x_end})")
    print(f"  {'TOTAL':>6s}: {total:3d} balls")
    print(f"{'='*40}")

    # Save annotated image
    display = frame.copy()
    colors = {"red": (0, 0, 180), "middle": (180, 180, 0), "blue": (180, 0, 0)}
    for name, (x_start, x_end) in zones.items():
        count = count_balls_in_region(yellow, x_start, x_end)
        # Tinted overlay
        overlay = display.copy()
        cv2.rectangle(overlay, (x_start, 0), (x_end, h), colors.get(name, (128, 128, 128)), -1)
        cv2.addWeighted(overlay, 0.15, display, 0.85, 0, display)
        # Zone boundary
        cv2.line(display, (x_start, 0), (x_start, h), (0, 255, 255), 2)
        cv2.line(display, (x_end, 0), (x_end, h), (0, 255, 255), 2)
        # Label
        cx = (x_start + x_end) // 2
        cv2.putText(display, f"{name.upper()}: {count}",
                    (cx - 80, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 4)

    cv2.imwrite("samples/field_counted.png", display)
    print("\nSaved annotated image to samples/field_counted.png")


if __name__ == "__main__":
    main()
