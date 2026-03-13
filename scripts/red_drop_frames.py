"""Grab frames from red goal video around scoring events to see the drop path."""

import json
import cv2

with open("samples/red/sample1-goal.scores.json") as f:
    scores = json.load(f)

cap = cv2.VideoCapture("samples/red/sample1-goal.mp4")

# Grab frames around the first few scoring events
# to see the ball drop trajectory
for score_frame in scores[:5]:
    for offset in [-10, -5, 0, 5, 10]:
        idx = score_frame + offset
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            path = f"samples/red/drop_f{idx:04d}.png"
            cv2.imwrite(path, frame)

    print(f"Saved frames around score at {score_frame}")

cap.release()
