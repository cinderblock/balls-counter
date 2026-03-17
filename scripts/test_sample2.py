"""Test motion counter on sample2-full with annotated geometry and ground truth."""

import json
import sys

sys.path.insert(0, "src")

import cv2
from ball_counter.counter import MotionCounter

VIDEO = "samples/red/sample2-full.mp4"
GEOM_FILE = "samples/red/sample2-full.geometry.json"
SCORES_FILE = "samples/red/sample2-full.scores.json"

with open(GEOM_FILE) as f:
    geometries = json.load(f)
with open(SCORES_FILE) as f:
    gt_scores = json.load(f)

cap = cv2.VideoCapture(VIDEO)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)

print(f"Video: {w}x{h} @ {fps}fps, {total} frames")
print(f"Ground truth: {len(gt_scores)} scores")
print(f"Geometries: {len(geometries)}")
print()

# Test each geometry
for geom in geometries:
    gtype = geom["type"]
    name = geom.get("name", gtype)
    pts = geom["points"]

    # Test a range of ball_area values
    for ba in [500, 700, 900, 1100, 1500, 2000, 3000, 5000]:
        for bw in [15, 20, 30, 40]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            if gtype == "line":
                counter = MotionCounter(
                    frame_shape=(h, w),
                    line=tuple(pts),
                    ball_area=ba,
                    band_width=bw,
                )
            else:
                counter = MotionCounter(
                    frame_shape=(h, w),
                    roi=pts,
                    ball_area=ba,
                    band_width=bw,
                )

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                counter.process_frame(frame)

            # Match events to ground truth (±15 frame tolerance)
            matched = 0
            used_gt = set()
            for ev in counter.events:
                for i, gt in enumerate(gt_scores):
                    if i not in used_gt and abs(ev.frame - gt) <= 15:
                        matched += 1
                        used_gt.add(i)
                        break

            fp = len(counter.events) - matched
            missed = len(gt_scores) - matched

            if matched >= 25 or (matched >= 20 and fp <= 5):
                print(f"  {name} ba={ba:5d} bw={bw:2d}: count={counter.count:3d} events={len(counter.events):3d} "
                      f"matched={matched:2d}/{len(gt_scores)} FP={fp:2d} missed={missed:2d}")

print("\nDone")
