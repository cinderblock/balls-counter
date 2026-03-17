"""Analyze peak areas to find the right ball_area divisor for sample2."""

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

cap = cv2.VideoCapture(VIDEO)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Run with a very small ball_area so n_balls reflects the raw peak
counter = MotionCounter(
    frame_shape=(h, w),
    line=tuple(line_pts),
    ball_area=900,
    band_width=15,
)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    counter.process_frame(frame)

cap.release()

print(f"Events: {len(counter.events)}, GT: {len(gt_scores)}")
print()

# Show each event with its peak area and which GT frames it's near
for ev in counter.events:
    nearby_gt = [g for g in gt_scores if abs(ev.frame - g) <= 15]
    # How many GT scores are near this event?
    print(f"  Frame {ev.frame:5d}  peak={ev.peak_area:6d}  n_balls={ev.n_balls}  "
          f"nearby_GT={nearby_gt} ({len(nearby_gt)} scores)")

print()
print("Missed GT frames and their nearest events:")
missed = [213, 238, 876, 882, 883, 885, 899, 904, 1497, 1513]
for m in missed:
    nearest_ev = min(counter.events, key=lambda e: abs(e.frame - m))
    print(f"  GT {m:5d}  nearest_event: frame={nearest_ev.frame} peak={nearest_ev.peak_area} "
          f"n_balls={nearest_ev.n_balls} (delta={abs(nearest_ev.frame - m)})")
