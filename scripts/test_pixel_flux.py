"""Pixel flux counter: measure yellow pixel flow across a line.

Instead of tracking individual balls, measure how much new yellow
appears below the line each frame. Accumulate until it exceeds
one ball's worth of pixels, then count a score.
"""

import json
import sys

import cv2
import numpy as np

sys.path.insert(0, "src")
from ball_counter.detector import create_mask

with open("samples/red/sample1-goal.line.json") as f:
    line_data = json.load(f)
p1, p2 = line_data["line"]

with open("samples/red/sample1-goal-drops2.scores.json") as f:
    ground_truth = json.load(f)

cap = cv2.VideoCapture("samples/red/sample1-goal.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)

# Create a mask for "below the line" region
ret, first_frame = cap.read()
h, w = first_frame.shape[:2]

# Build a mask for the region below the counting line
below_mask = np.zeros((h, w), dtype=np.uint8)
# The line goes from p1 to p2. "Below" = the side where balls drop to.
# Fill the polygon below the line to the bottom of frame
line_pts = np.array([p1, p2, [w, h], [0, h]], dtype=np.int32)
cv2.fillPoly(below_mask, [line_pts], 255)

# Also make a narrow band mask right around the line (±20px)
band_mask = np.zeros((h, w), dtype=np.uint8)
band_above = np.array([
    [p1[0], p1[1] - 25], [p2[0], p2[1] - 25],
    [p2[0], p2[1] + 25], [p1[0], p1[1] + 25]
], dtype=np.int32)
cv2.fillPoly(band_mask, [band_above], 255)

# Reset to start
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

prev_mask = None
accumulated_area = 0.0
count = 0
frame_idx = 0
events = []

# Calibrate ball area from an isolated ball (estimated)
BALL_AREA_PX = 700  # approximate yellow pixels per ball

while True:
    ret, frame = cap.read()
    if not ret:
        break

    yellow = create_mask(frame)

    # Yellow pixels in the band around the line
    yellow_in_band = cv2.bitwise_and(yellow, band_mask)

    if prev_mask is not None:
        # New yellow pixels that appeared in the band this frame
        new_yellow = cv2.bitwise_and(yellow_in_band, cv2.bitwise_not(prev_mask))
        new_area = cv2.countNonZero(new_yellow)

        # Only accumulate significant new area (filter noise)
        if new_area > 30:
            accumulated_area += new_area

        # Check if we've accumulated enough for a ball
        while accumulated_area >= BALL_AREA_PX * 0.5:
            count += 1
            accumulated_area -= BALL_AREA_PX * 0.5
            events.append(frame_idx)
            print(f"  Frame {frame_idx:4d} ({frame_idx/fps:6.2f}s): SCORED! Total: {count}")

    prev_mask = yellow_in_band
    frame_idx += 1

cap.release()

print(f"\n--- Results ---")
print(f"Pixel flux count: {count}")
print(f"Ground truth: {len(ground_truth)}")
print(f"Difference: {abs(count - len(ground_truth))}")
