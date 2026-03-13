"""Test motion counter using a thin ring band around a polygon ROI perimeter."""

import json
import sys

import cv2
import numpy as np

sys.path.insert(0, "src")
from ball_counter.detector import create_mask


def make_ring_mask(h, w, roi_pts, band_width=20):
    """Create a ring-shaped mask along the polygon perimeter."""
    # Outer polygon (expanded)
    outer = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(outer, [roi_pts], 255)
    outer_dilated = cv2.dilate(outer, np.ones((band_width, band_width), np.uint8))

    # Inner polygon (shrunk)
    inner_eroded = cv2.erode(outer, np.ones((band_width, band_width), np.uint8))

    # Ring = dilated minus eroded
    ring = cv2.subtract(outer_dilated, inner_eroded)
    return ring


def run_motion_ring(video_path, roi_json_path, ball_area=900, band_width=15,
                    min_peak=0, fall_ratio=0.5):
    with open(roi_json_path) as f:
        roi_data = json.load(f)
    roi_pts = np.array(roi_data["roi"], dtype=np.int32)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    ret, first = cap.read()
    h, w = first.shape[:2]

    ring_mask = make_ring_mask(h, w, roi_pts, band_width)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    bg_sub = cv2.createBackgroundSubtractorMOG2(
        history=60, varThreshold=50, detectShadows=False,
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    events = []
    frame_idx = 0
    prev_area = 0
    rising = False
    peak_val = 0
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        fg_mask = bg_sub.apply(frame)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

        yellow = create_mask(frame)
        moving_yellow = cv2.bitwise_and(yellow, fg_mask)
        moving_in_ring = cv2.bitwise_and(moving_yellow, ring_mask)
        area = cv2.countNonZero(moving_in_ring)

        if area > prev_area and area > 100:
            rising = True
            peak_val = max(peak_val, area)
        elif rising and area < peak_val * fall_ratio:
            if peak_val >= min_peak:
                n_balls = max(1, round(peak_val / ball_area))
                count += n_balls
                events.append((frame_idx, n_balls, peak_val))
            rising = False
            peak_val = 0

        prev_area = area
        frame_idx += 1

    cap.release()
    return events, fps, count


def compare_events(events, ground_truth, tolerance_frames=15):
    detected_frames = []
    for frame, n_balls, peak in events:
        for _ in range(n_balls):
            detected_frames.append(frame)

    gt = sorted(ground_truth)
    det = sorted(detected_frames)

    matched_gt = set()
    matched_det = set()

    for di, df in enumerate(det):
        best_dist = tolerance_frames + 1
        best_gi = -1
        for gi, gf in enumerate(gt):
            if gi in matched_gt:
                continue
            dist = abs(df - gf)
            if dist < best_dist:
                best_dist = dist
                best_gi = gi
        if best_gi >= 0 and best_dist <= tolerance_frames:
            matched_gt.add(best_gi)
            matched_det.add(di)

    missed = [gt[i] for i in range(len(gt)) if i not in matched_gt]
    false_pos = [det[i] for i in range(len(det)) if i not in matched_det]
    return len(matched_gt), missed, false_pos


with open("samples/blue/sample1-goal.scores.json") as f:
    gt_blue = json.load(f)

print(f"Blue goal - Ground truth: {len(gt_blue)} events\n")

print("Ring band approach:")
for ball_area in [400, 500, 600, 700, 900, 1100]:
    for band_width in [10, 15, 20, 25]:
        for min_peak in [0, 200, 400]:
            events, fps, count = run_motion_ring(
                "samples/blue/sample1-goal.mp4",
                "samples/blue/sample1-goal.roi.json",
                ball_area=ball_area,
                band_width=band_width,
                min_peak=min_peak,
            )
            total = sum(n for _, n, _ in events)
            matched, missed, fp = compare_events(events, gt_blue)

            if abs(total - len(gt_blue)) <= 5 or matched >= 28:
                print(f"  ba={ball_area:4d} bw={band_width:2d} mp={min_peak:3d}: "
                      f"total={total:3d} match={matched:2d} miss={len(missed):2d} fp={len(fp):2d}")

# Show best default
print(f"\n{'='*60}")
print("Detailed run (ba=700, bw=15, mp=0):")
events, fps, count = run_motion_ring(
    "samples/blue/sample1-goal.mp4",
    "samples/blue/sample1-goal.roi.json",
    ball_area=700,
    band_width=15,
)
print(f"\nDetected {len(events)} events, total count={count}:")
for frame, n_balls, peak in events:
    print(f"  Frame {frame:4d} ({frame/fps:6.2f}s): +{n_balls} (peak={peak})")
matched, missed, fp = compare_events(events, gt_blue)
print(f"\nMatched: {matched}/{len(gt_blue)}")
print(f"Missed:  {missed}")
print(f"False+:  {fp}")
