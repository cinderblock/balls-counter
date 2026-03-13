"""Motion-based ball counter: detect new yellow blobs crossing the line.

Use frame differencing to detect actual ball motion, not static pile shifts.
Only count yellow blobs that are both:
  1. Moving (appear in frame diff)
  2. Crossing the counting line
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

VIDEO_PATH = "samples/red/sample1-goal.mp4"
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)

# Narrow band around the line
ret, first = cap.read()
h, w = first.shape[:2]
band_mask = np.zeros((h, w), dtype=np.uint8)
band_pts = np.array([
    [p1[0], p1[1] - 20], [p2[0], p2[1] - 20],
    [p2[0], p2[1] + 20], [p1[0], p1[1] + 20]
], dtype=np.int32)
cv2.fillPoly(band_mask, [band_pts], 255)

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Background subtractor — learns what's static
bg_sub = cv2.createBackgroundSubtractorMOG2(
    history=60,
    varThreshold=50,
    detectShadows=False,
)

count = 0
frame_idx = 0
accumulated = 0.0
BALL_AREA = 900

# Smoothed yellow-in-band signal for peak detection
prev_yellow_in_band = 0
rising = False
peak_val = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Get motion mask
    fg_mask = bg_sub.apply(frame)
    # Clean up motion mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

    # Get yellow mask
    yellow = create_mask(frame)

    # Moving yellow in the band = ball actively crossing
    moving_yellow = cv2.bitwise_and(yellow, fg_mask)
    moving_in_band = cv2.bitwise_and(moving_yellow, band_mask)
    area = cv2.countNonZero(moving_in_band)

    # Peak detection: score when moving-yellow-in-band rises then falls
    if area > prev_yellow_in_band and area > 100:
        rising = True
        peak_val = max(peak_val, area)
    elif rising and area < peak_val * 0.5:
        # Peak ended — count balls based on peak area
        n_balls = max(1, round(peak_val / BALL_AREA))
        count += n_balls
        print(f"  Frame {frame_idx:4d} ({frame_idx/fps:6.2f}s): +{n_balls} (peak={peak_val}) Total: {count}")
        rising = False
        peak_val = 0

    prev_yellow_in_band = area
    frame_idx += 1

cap.release()

print(f"\n--- Results ---")
print(f"Motion counter: {count}")
print(f"Ground truth: {len(ground_truth)}")
print(f"Difference: {abs(count - len(ground_truth))}")
