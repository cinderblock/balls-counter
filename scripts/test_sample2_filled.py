"""Test filled polygon (interior) approach on sample2."""

import json
import sys

sys.path.insert(0, "src")

import cv2
import numpy as np
from ball_counter.detector import create_mask

VIDEO = "samples/blue/sample2-full.mp4"
GEOM_FILE = "samples/blue/sample2-full.geometry.json"
SCORES_FILE = "samples/blue/sample2-full.scores.json"

with open(GEOM_FILE) as f:
    geometries = json.load(f)
with open(SCORES_FILE) as f:
    gt_scores = json.load(f)

cap = cv2.VideoCapture(VIDEO)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

poly = None
for g in geometries:
    if g["type"] == "polygon":
        poly = g["points"]
        break

if not poly:
    print("No polygon found")
    sys.exit(1)

# Build filled polygon mask
poly_mask = np.zeros((h, w), dtype=np.uint8)
cv2.fillPoly(poly_mask, [np.array(poly, dtype=np.int32)], 255)

# Also try expanded versions (scale polygon outward from centroid)
cx = sum(p[0] for p in poly) / len(poly)
cy = sum(p[1] for p in poly) / len(poly)

for scale in [1.0, 1.2, 1.5, 2.0]:
    for ba in [500, 900, 1500, 3000, 5000]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # Scale polygon
        scaled_poly = []
        for p in poly:
            sx = int(cx + (p[0] - cx) * scale)
            sy = int(cy + (p[1] - cy) * scale)
            scaled_poly.append([sx, sy])

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(scaled_poly, dtype=np.int32)], 255)

        bg_sub = cv2.createBackgroundSubtractorMOG2(
            history=60, varThreshold=50, detectShadows=False
        )
        morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        prev_area = 0
        rising = False
        peak_val = 0
        events = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            fg = bg_sub.apply(frame)
            fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, morph_kernel)
            yellow = create_mask(frame)
            moving_yellow = cv2.bitwise_and(yellow, fg)
            in_zone = cv2.bitwise_and(moving_yellow, mask)
            area = cv2.countNonZero(in_zone)

            if area > prev_area and area > 100:
                rising = True
                peak_val = max(peak_val, area)
            elif rising and area < peak_val * 0.5:
                if peak_val >= 100:
                    n = max(1, round(peak_val / ba))
                    events.append((frame_idx, n, peak_val))
                rising = False
                peak_val = 0

            prev_area = area
            frame_idx += 1

        total_count = sum(e[1] for e in events)

        # Match
        matched_gt = set()
        matched_ev = set()
        for i, ev in enumerate(events):
            for j, gt in enumerate(gt_scores):
                if j not in matched_gt and abs(ev[0] - gt) <= 15:
                    matched_gt.add(j)
                    matched_ev.add(i)
                    break

        fp = len(events) - len(matched_ev)
        missed = len(gt_scores) - len(matched_gt)

        marker = ""
        if len(matched_gt) >= 30 and fp <= 5:
            marker = " ***"
        elif len(matched_gt) >= 25:
            marker = " **"
        elif len(matched_gt) >= 20:
            marker = " *"

        print(f"  filled scale={scale:.1f} ba={ba:5d}: count={total_count:3d} events={len(events):3d} "
              f"matched={len(matched_gt):2d}/{len(gt_scores)} FP={fp:2d} missed={missed:2d}{marker}")

        if len(matched_gt) >= 25 and missed > 0:
            missed_frames = [gt_scores[j] for j in range(len(gt_scores)) if j not in matched_gt]
            print(f"    Missed: {missed_frames}")

print("\nDone")
