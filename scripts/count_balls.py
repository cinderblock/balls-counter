"""How many balls does OpenCV see? Break down the detection pipeline."""

import cv2
import numpy as np
import sys

sys.path.insert(0, "src")
from ball_counter.detector import create_mask

cap = cv2.VideoCapture("samples/red/sample1-goal.mp4")

for target in [350, 430, 500]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, target)
    ret, frame = cap.read()
    if not ret:
        continue

    mask = create_mask(frame)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    total_contours = len(contours)
    yellow_pixels = cv2.countNonZero(mask)

    # Categorize contours
    tiny = []      # < 100 px (noise)
    small = []     # 100-500 px
    medium = []    # 500-3000 px (single ball range)
    large = []     # > 3000 px (merged clusters)

    for c in contours:
        area = cv2.contourArea(c)
        if area < 100:
            tiny.append(area)
        elif area < 500:
            small.append(area)
        elif area < 3000:
            medium.append(area)
        else:
            large.append(area)

    # Estimate single ball area from medium contours
    if medium:
        median_ball = sorted(medium)[len(medium) // 2]
    else:
        median_ball = 600  # fallback

    # Estimate balls in large clusters
    cluster_balls = sum(int(a / median_ball) for a in large)

    print(f"\n--- Frame {target} ---")
    print(f"  Yellow pixels: {yellow_pixels}")
    print(f"  Total contours: {total_contours}")
    print(f"  Tiny (<100px, noise): {len(tiny)}")
    print(f"  Small (100-500px): {len(small)}")
    print(f"  Medium (500-3000px, likely single balls): {len(medium)}")
    print(f"  Large (>3000px, merged clusters): {len(large)}")
    if large:
        print(f"    Cluster areas: {[int(a) for a in large]}")
        print(f"    Est. balls in clusters: ~{cluster_balls} (using {median_ball:.0f}px/ball)")
    print(f"  Total estimate: ~{len(medium) + cluster_balls} balls")

    # Circularity check on medium contours
    circular = 0
    for c in contours:
        area = cv2.contourArea(c)
        if area < 500:
            continue
        perimeter = cv2.arcLength(c, True)
        if perimeter == 0:
            continue
        circ = 4 * np.pi * area / (perimeter * perimeter)
        if circ >= 0.5 and area < 3000:
            circular += 1

    print(f"  Pass circularity filter (what detect_balls returns): {circular}")

cap.release()
