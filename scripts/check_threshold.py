"""Check HSV threshold quality on red goal frames."""

import cv2
import numpy as np
import sys

sys.path.insert(0, "src")
from ball_counter.detector import create_mask, detect_balls

cap = cv2.VideoCapture("samples/red/sample1-goal.mp4")

for target in [400, 430, 460, 500]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, target)
    ret, frame = cap.read()
    if not ret:
        continue

    mask = create_mask(frame)
    detections = detect_balls(frame)

    # Side by side: original with detections | mask
    display = frame.copy()
    for d in detections:
        cx, cy = d["center"]
        cv2.circle(display, (cx, cy), d["radius"], (0, 255, 0), 2)

    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    combined = np.hstack([display, mask_bgr])

    path = f"samples/red/threshold_f{target:04d}.png"
    cv2.imwrite(path, combined)
    print(f"Frame {target}: {len(detections)} detections -> {path}")

cap.release()
