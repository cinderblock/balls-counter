"""Test both detection approaches on the red goal frames."""

import cv2
import numpy as np
import sys

sys.path.insert(0, "src")
from ball_counter.detector import detect_balls, create_mask

for name in ["red_frame_00", "red_frame_50", "red_frame_75"]:
    frame = cv2.imread(f"{name}.png")
    if frame is None:
        continue

    # Individual ball detection
    detections = detect_balls(frame)

    # Area-based estimation
    mask = create_mask(frame)
    yellow_pixels = cv2.countNonZero(mask)

    # Estimate single ball area from isolated detections
    # Use the median area of detected balls as reference
    if detections:
        areas = sorted(d["area"] for d in detections)
        # Use smaller detections as they're more likely to be single balls
        single_ball_area = np.median(areas[:max(1, len(areas) // 2)])
        area_estimate = int(yellow_pixels / single_ball_area) if single_ball_area > 0 else 0
    else:
        single_ball_area = 0
        area_estimate = 0

    # Draw both results
    display = frame.copy()
    for d in detections:
        cx, cy = d["center"]
        r = d["radius"]
        cv2.circle(display, (cx, cy), r, (0, 255, 0), 2)

    cv2.putText(
        display,
        f"Individual detect: {len(detections)} balls",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        display,
        f"Area estimate: ~{area_estimate} balls ({yellow_pixels} px / {single_ball_area:.0f} px/ball)",
        (10, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 255),
        2,
    )

    out_path = f"detect2_{name}.png"
    cv2.imwrite(out_path, display)
    print(f"{name}: detect={len(detections)}, area_est=~{area_estimate} (yellow_px={yellow_pixels}, ball_area={single_ball_area:.0f})")
