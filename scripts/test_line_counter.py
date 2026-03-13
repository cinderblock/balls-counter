"""Test line-crossing counter on the red goal drop sample.

Runs detection + tracking + line crossing and compares to ground truth.
"""

import json
import sys

import cv2

sys.path.insert(0, "src")
from ball_counter.counter import LineCrossingCounter
from ball_counter.detector import detect_balls
from ball_counter.tracker import CentroidTracker

# Load line
with open("samples/red/sample1-goal.line.json") as f:
    line_data = json.load(f)
p1, p2 = line_data["line"]

# Load ground truth
with open("samples/red/sample1-goal-drops2.scores.json") as f:
    ground_truth = json.load(f)

counting_line = LineCrossingCounter(x1=p1[0], y1=p1[1], x2=p2[0], y2=p2[1])
tracker = CentroidTracker(max_disappeared=30)

cap = cv2.VideoCapture("samples/red/sample1-goal.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
frame_idx = 0
events = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    detections = detect_balls(frame)
    centroids = [d["center"] for d in detections]
    objects = tracker.update(centroids)

    for object_id, centroid in objects.items():
        event = counting_line.update_object(object_id, int(centroid[0]), int(centroid[1]))
        if event:
            events.append({
                "frame": frame_idx,
                "time": frame_idx / fps,
                "direction": event,
                "object_id": object_id,
            })
            print(f"  Frame {frame_idx:4d} ({frame_idx/fps:6.2f}s): {event.upper()} (ID {object_id})")

    counting_line.cleanup(set(objects.keys()))
    frame_idx += 1

cap.release()

in_count = counting_line.count_in
out_count = counting_line.count_out
print(f"\n--- Results ---")
print(f"IN:  {in_count}")
print(f"OUT: {out_count}")
print(f"NET: {in_count - out_count}")
print(f"Ground truth drops: {len(ground_truth)}")
print(f"Difference: {abs((in_count - out_count) - len(ground_truth))}")
