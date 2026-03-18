"""Diagnostic: dump raw area signal and show where GT events fall.

Outputs per-frame signal values so we can understand why bursts merge.
Also tests: what if instead of peak detection on raw area,
we count individual yellow blobs (connected components) in the zone?
"""

import json
import sys

sys.path.insert(0, "src")

import cv2
import numpy as np
from ball_counter.detector import create_mask

PADDING = 150


def run_diagnosis(video_path, geom_path, scores_path, line_indices):
    with open(geom_path) as f:
        geometries = json.load(f)
    with open(scores_path) as f:
        gt_scores = json.load(f)

    lines = [g for g in geometries if g["type"] == "line"]
    use_lines = [lines[i] for i in line_indices]

    cap = cv2.VideoCapture(video_path)
    full_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    full_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Compute crop
    all_xs, all_ys = [], []
    for lg in use_lines:
        for p in lg["points"]:
            all_xs.append(p[0])
            all_ys.append(p[1])
    x1 = max(0, min(all_xs) - PADDING)
    y1 = max(0, min(all_ys) - PADDING)
    x2 = min(full_w, max(all_xs) + PADDING)
    y2 = min(full_h, max(all_ys) + PADDING)
    ch, cw = y2 - y1, x2 - x1

    # Cache frames
    print(f"Caching {total_frames} frames at {cw}x{ch}...", flush=True)
    frames = []
    for _ in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame[y1:y2, x1:x2].copy())
    cap.release()
    print(f"Cached {len(frames)} frames")

    # Build band mask (bw=15, a mid-range value)
    bw = 15
    band_mask = np.zeros((ch, cw), dtype=np.uint8)
    for lg in use_lines:
        p1, p2 = lg["points"]
        cp1 = [p1[0] - x1, p1[1] - y1]
        cp2 = [p2[0] - x1, p2[1] - y1]
        dx = cp2[0] - cp1[0]
        dy = cp2[1] - cp1[1]
        length = np.sqrt(dx * dx + dy * dy)
        if length == 0:
            continue
        nx = -dy / length * bw
        ny = dx / length * bw
        pts = np.array([
            [cp1[0] + nx, cp1[1] + ny],
            [cp2[0] + nx, cp2[1] + ny],
            [cp2[0] - nx, cp2[1] - ny],
            [cp1[0] - nx, cp1[1] - ny],
        ], dtype=np.int32)
        cv2.fillPoly(band_mask, [pts], 255)

    # Process frames
    bg_sub = cv2.createBackgroundSubtractorMOG2(
        history=60, varThreshold=50, detectShadows=False
    )
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    signals = []
    blob_counts = []

    for frame in frames:
        fg = bg_sub.apply(frame)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, morph_kernel)
        yellow = create_mask(frame)
        moving_yellow = cv2.bitwise_and(yellow, fg)
        in_band = cv2.bitwise_and(moving_yellow, band_mask)
        area = cv2.countNonZero(in_band)
        signals.append(area)

        # Count separate blobs
        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            in_band, connectivity=8
        )
        # Exclude background (label 0), count blobs with area >= 50
        n_blobs = sum(1 for i in range(1, n_labels) if stats[i, cv2.CC_STAT_AREA] >= 50)
        blob_counts.append(n_blobs)

    # Print signal around GT events
    print(f"\n{'Frame':>6} {'Area':>6} {'Blobs':>5}  GT")
    print("-" * 35)

    gt_set = set(gt_scores)
    # Show frames around the densest burst regions
    interesting_start = max(0, gt_scores[0] - 10)
    interesting_end = min(len(signals), gt_scores[-1] + 20)

    for i in range(interesting_start, interesting_end):
        marker = " <-- SCORE" if i in gt_set else ""
        area = signals[i] if i < len(signals) else 0
        blobs = blob_counts[i] if i < len(blob_counts) else 0
        if area > 0 or marker or (i > 0 and signals[i - 1] > 0):
            print(f"{i:6d} {area:6d} {blobs:5d}  {marker}")

    # Summary stats
    gt_areas = [signals[f] for f in gt_scores if f < len(signals)]
    gt_blobs = [blob_counts[f] for f in gt_scores if f < len(blob_counts)]
    print(f"\nAt GT frames: area range {min(gt_areas)}-{max(gt_areas)}, "
          f"blob count range {min(gt_blobs)}-{max(gt_blobs)}")

    # Can blob counting solve it? Sum blobs at GT frames
    total_blobs_at_gt = sum(gt_blobs)
    print(f"Sum of blobs at GT frames: {total_blobs_at_gt} (want ~{len(gt_scores)})")

    # Max simultaneous blobs during any burst
    max_blobs_any = max(blob_counts[interesting_start:interesting_end])
    print(f"Max simultaneous blobs in active region: {max_blobs_any}")


print("=" * 60)
print("RED SAMPLE (line-1 only)")
print("=" * 60)
run_diagnosis(
    "samples/new-angle/sample-red.mp4",
    "samples/new-angle/sample-red.geometry.json",
    "samples/new-angle/sample-red.scores.json",
    [0],
)

print("\n" + "=" * 60)
print("BLUE SAMPLE (combined lines)")
print("=" * 60)
run_diagnosis(
    "samples/new-angle/sample-blue.mp4",
    "samples/new-angle/sample-blue.geometry.json",
    "samples/new-angle/sample-blue.scores.json",
    [0, 1],
)
