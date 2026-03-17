"""Quick sweep of ball_area on sample2 to find the best divisor."""

import json
import sys

sys.path.insert(0, "src")

import cv2
from ball_counter.counter import MotionCounter

VIDEO = "samples/blue/sample2-full.mp4"
GEOM_FILE = "samples/blue/sample2-full.geometry.json"
SCORES_FILE = "samples/blue/sample2-full.scores.json"

with open(GEOM_FILE) as f:
    geometries = json.load(f)
with open(SCORES_FILE) as f:
    gt_scores = json.load(f)

line_pts = None
for g in geometries:
    if g["type"] == "line":
        line_pts = g["points"]
        break

cap_ref = cv2.VideoCapture(VIDEO)
w = int(cap_ref.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap_ref.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap_ref.release()

for ba in [200, 300, 350, 400, 450, 500, 550, 600, 700, 800]:
    for bw in [12, 15, 20, 25]:
        cap = cv2.VideoCapture(VIDEO)
        counter = MotionCounter(
            frame_shape=(h, w),
            line=tuple(line_pts),
            ball_area=ba,
            band_width=bw,
        )

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            counter.process_frame(frame)
        cap.release()

        # Match events to GT
        matched_gt = set()
        matched_ev = set()
        for i, ev in enumerate(counter.events):
            for j, gt_f in enumerate(gt_scores):
                if j not in matched_gt and abs(ev.frame - gt_f) <= 15:
                    matched_gt.add(j)
                    matched_ev.add(i)
                    break

        fp = len(counter.events) - len(matched_ev)
        total_count = counter.count

        marker = ""
        if total_count == len(gt_scores) and fp == 0:
            marker = " *** PERFECT"
        elif abs(total_count - len(gt_scores)) <= 2 and fp <= 2:
            marker = " **"
        elif abs(total_count - len(gt_scores)) <= 5:
            marker = " *"

        print(f"ba={ba:4d} bw={bw:2d}: events={len(counter.events):3d} "
              f"count={total_count:3d} matched={len(matched_gt):2d}/{len(gt_scores)} "
              f"FP={fp:2d}{marker}")

print("\nDone")
