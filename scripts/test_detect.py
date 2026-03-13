"""Test detection on sample frames — left half only."""

import cv2
import numpy as np
import sys

sys.path.insert(0, "src")
from ball_counter.detector import detect_balls

for name in ["frame_00", "frame_25", "frame_50", "frame_75", "frame_90"]:
    frame = cv2.imread(f"{name}.png")
    if frame is None:
        continue

    # Crop to left half
    h, w = frame.shape[:2]
    frame = frame[:, : w // 2]

    detections = detect_balls(frame)

    # Draw detections
    for d in detections:
        cx, cy = d["center"]
        r = d["radius"]
        cv2.circle(frame, (cx, cy), r, (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)

    cv2.putText(
        frame,
        f"Detected: {len(detections)}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
    )

    out_path = f"detect_{name}.png"
    cv2.imwrite(out_path, frame)
    print(f"{name}: {len(detections)} balls detected -> {out_path}")
