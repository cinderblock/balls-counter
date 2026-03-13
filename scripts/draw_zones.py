"""Draw vertical zone boundaries on the field snapshot.

Click to place two vertical dividers splitting the field into 3 zones:
  Red (left) | Middle | Blue (right)

Drag lines to adjust. Enter to save. Shows yellow ball count per zone.
"""

import json
import sys

import cv2
import numpy as np

sys.path.insert(0, "src")
from ball_counter.detector import create_mask

snapshot_path = "samples/field_snapshot.png"
output_path = "samples/field_zones.json"

frame = cv2.imread(snapshot_path)
if frame is None:
    print(f"Error: cannot read {snapshot_path}", file=sys.stderr)
    sys.exit(1)

h, w = frame.shape[:2]
print(f"Image: {w}x{h}")

# Pre-compute yellow mask
yellow = create_mask(frame)

# Initial dividers at 1/3 and 2/3
dividers = [w // 3, 2 * w // 3]
dragging = -1  # which divider is being dragged

window = "Zone Setup - drag lines, Enter=save, Q=quit"
cv2.namedWindow(window, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window, min(w, 1920), min(h, 1080))


def count_in_zone(mask, x_start, x_end):
    zone = mask[:, x_start:x_end]
    # Count blobs, not just pixels
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        zone, connectivity=8)
    count = 0
    for i in range(1, n_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= 100:  # filter noise
            count += 1
    total_px = cv2.countNonZero(zone)
    return count, total_px


def redraw():
    display = frame.copy()

    # Draw zone overlays
    zones = [
        (0, dividers[0], (0, 0, 180), "RED"),
        (dividers[0], dividers[1], (180, 180, 0), "MID"),
        (dividers[1], w, (180, 0, 0), "BLUE"),
    ]

    for x_start, x_end, color, label in zones:
        overlay = display.copy()
        cv2.rectangle(overlay, (x_start, 0), (x_end, h), color, -1)
        cv2.addWeighted(overlay, 0.15, display, 0.85, 0, display)

        balls, px = count_in_zone(yellow, x_start, x_end)
        cx = (x_start + x_end) // 2
        cv2.putText(display, f"{label}: {balls} balls",
                    (cx - 80, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 4)
        cv2.putText(display, f"({px} px)",
                    (cx - 60, h // 2 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

    # Draw divider lines
    for x in dividers:
        cv2.line(display, (x, 0), (x, h), (0, 255, 255), 3)

    # Instructions
    cv2.rectangle(display, (0, 0), (w, 40), (0, 0, 0), -1)
    cv2.putText(display, f"Dividers at x={dividers[0]}, x={dividers[1]} | Drag to adjust | Enter=save  Q=quit",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow(window, display)


def on_mouse(event, x, y, flags, param):
    global dragging

    if event == cv2.EVENT_LBUTTONDOWN:
        # Check if near a divider
        for i, dx in enumerate(dividers):
            if abs(x - dx) < 30:
                dragging = i
                break
    elif event == cv2.EVENT_MOUSEMOVE and dragging >= 0:
        dividers[dragging] = max(50, min(w - 50, x))
        # Keep dividers ordered
        if dividers[0] > dividers[1] - 50:
            if dragging == 0:
                dividers[0] = dividers[1] - 50
            else:
                dividers[1] = dividers[0] + 50
        redraw()
    elif event == cv2.EVENT_LBUTTONUP:
        dragging = -1


cv2.setMouseCallback(window, on_mouse)
redraw()

while True:
    key = cv2.waitKey(50) & 0xFF
    if key == ord("q"):
        break
    elif key == 13:
        data = {
            "zones": {
                "red": [0, dividers[0]],
                "middle": [dividers[0], dividers[1]],
                "blue": [dividers[1], w],
            },
            "image_size": [w, h],
        }
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved zones to {output_path}")
        print(f"  Red:    0 - {dividers[0]}")
        print(f"  Middle: {dividers[0]} - {dividers[1]}")
        print(f"  Blue:   {dividers[1]} - {w}")
        break

cv2.destroyAllWindows()
