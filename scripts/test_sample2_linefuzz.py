"""Fuzz line position on sample2 to find optimal placement."""

import json
import math
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

p1, p2 = line_pts
mx = (p1[0] + p2[0]) / 2
my = (p1[1] + p2[1]) / 2
dx = p2[0] - p1[0]
dy = p2[1] - p1[1]
length = math.sqrt(dx * dx + dy * dy)
angle = math.atan2(dy, dx)

# Normal direction (perpendicular to line)
nx = -dy / length
ny = dx / length

results = []

# Sweep: shift along normal (up/down), shift along line direction, rotate, extend length
combos = 0
for shift_normal in range(-80, 81, 20):  # perpendicular shift (9 values)
    for shift_along in range(-60, 61, 20):  # parallel shift (7 values)
        for angle_offset in [0]:  # no rotation
            for len_mult in [0.8, 1.0, 1.3, 1.6, 2.0]:  # length (5 values)
                new_angle = angle + angle_offset
                new_mx = mx + nx * shift_normal + (dx / length) * shift_along
                new_my = my + ny * shift_normal + (dy / length) * shift_along
                half_len = (length * len_mult) / 2

                np1 = [int(new_mx - math.cos(new_angle) * half_len),
                       int(new_my - math.sin(new_angle) * half_len)]
                np2 = [int(new_mx + math.cos(new_angle) * half_len),
                       int(new_my + math.sin(new_angle) * half_len)]

                combos += 1
                for ba in [400, 500, 600]:
                    cap = cv2.VideoCapture(VIDEO)
                    try:
                        counter = MotionCounter(
                            frame_shape=(h, w),
                            line=(np1, np2),
                            ball_area=ba,
                            band_width=15,
                            fall_ratio=0.7,
                            min_rise=50,
                        )
                    except Exception:
                        cap.release()
                        continue

                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        counter.process_frame(frame)
                    cap.release()

                    matched_gt = set()
                    matched_ev = set()
                    for i, ev in enumerate(counter.events):
                        for j, gt_f in enumerate(gt_scores):
                            if j not in matched_gt and abs(ev.frame - gt_f) <= 15:
                                matched_gt.add(j)
                                matched_ev.add(i)
                                break

                    fp = len(counter.events) - len(matched_ev)
                    matched = len(matched_gt)
                    count = counter.count

                    results.append((matched, fp, count, shift_normal, shift_along,
                                    angle_offset, len_mult, ba, np1, np2))

                    if combos % 50 == 0:
                        print(f"  ... {combos} combos tested, best so far: {max((r[0] for r in results), default=0)}/{len(gt_scores)} matched", flush=True)

                    if matched >= 30 and fp <= 3:
                        print(f"*** matched={matched}/{len(gt_scores)} FP={fp} count={count} "
                              f"shift_n={shift_normal} shift_a={shift_along} "
                              f"angle={angle_offset:+.2f} len={len_mult:.1f} ba={ba}")

# Sort by matched desc, then FP asc, then abs(count-35)
results.sort(key=lambda r: (-r[0], r[1], abs(r[2] - len(gt_scores))))

print(f"\nTop 20 results:")
for r in results[:20]:
    matched, fp, count, sn, sa, ao, lm, ba, np1, np2 = r
    print(f"  matched={matched:2d}/{len(gt_scores)} FP={fp:2d} count={count:3d} "
          f"shift_n={sn:+4d} shift_a={sa:+4d} angle={ao:+.2f} len={lm:.1f} ba={ba} "
          f"line={np1}->{np2}")

print("\nDone")
