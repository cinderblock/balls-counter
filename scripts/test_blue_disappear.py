"""Disappearance-based counter for top-down goal view.

When a ball drops into the hole, it vanishes from the scene.
Track total yellow pixels in a region around the goal.
A sudden decrease = ball scored.
"""

import json
import sys

import cv2
import numpy as np

sys.path.insert(0, "src")
from ball_counter.detector import create_mask


def compare_events(events, ground_truth, tolerance_frames=15):
    detected_frames = []
    for frame, n_balls, *_ in events:
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


def run_disappear(video_path, roi_json_path, ball_area=700, region_pad=40,
                  min_drop=200, smoothing=3, cooldown=0):
    """Track yellow pixel count in a padded region around the ROI.
    When the count drops sharply, balls have disappeared into the hole."""
    with open(roi_json_path) as f:
        roi_pts = np.array(json.load(f)["roi"], dtype=np.int32)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    ret, first = cap.read()
    h, w = first.shape[:2]

    # Region mask: padded area around the ROI (where balls approach)
    region_mask = np.zeros((h, w), dtype=np.uint8)
    # Dilate the ROI polygon to include the approach area
    roi_filled = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(roi_filled, [roi_pts], 255)
    region_mask = cv2.dilate(roi_filled, np.ones((region_pad * 2, region_pad * 2), np.uint8))

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    events = []
    frame_idx = 0
    history = []
    cooldown_remaining = 0
    count = 0

    # Track the yellow pixel count over time
    prev_smoothed = 0
    falling = False
    trough_val = float('inf')
    pre_fall_val = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        yellow = create_mask(frame)
        yellow_in_region = cv2.bitwise_and(yellow, region_mask)
        area = cv2.countNonZero(yellow_in_region)

        # Smoothing: rolling average
        history.append(area)
        if len(history) > smoothing:
            history.pop(0)
        smoothed = sum(history) / len(history)

        if cooldown_remaining > 0:
            cooldown_remaining -= 1
            prev_smoothed = smoothed
            frame_idx += 1
            continue

        # Detect drops: smoothed signal decreasing significantly
        if smoothed < prev_smoothed - min_drop and not falling:
            falling = True
            pre_fall_val = prev_smoothed
            trough_val = smoothed
        elif falling and smoothed < trough_val:
            trough_val = smoothed
        elif falling and smoothed > trough_val + min_drop * 0.5:
            # Drop ended, count balls based on total drop magnitude
            drop = pre_fall_val - trough_val
            n_balls = max(1, round(drop / ball_area))
            count += n_balls
            events.append((frame_idx, n_balls, drop))
            cooldown_remaining = cooldown
            falling = False
            trough_val = float('inf')
            pre_fall_val = 0

        prev_smoothed = smoothed
        frame_idx += 1

    cap.release()
    return events, fps, count


# ============================================================
# Also try: delta-based (frame-to-frame yellow decrease in ROI)
# ============================================================
def run_delta_disappear(video_path, roi_json_path, ball_area=700,
                        region_pad=40, smoothing=3, cooldown=0):
    """Track frame-to-frame decrease in yellow pixels near the ROI.
    Accumulate decreases, score when enough yellow has vanished."""
    with open(roi_json_path) as f:
        roi_pts = np.array(json.load(f)["roi"], dtype=np.int32)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    ret, first = cap.read()
    h, w = first.shape[:2]

    roi_filled = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(roi_filled, [roi_pts], 255)
    region_mask = cv2.dilate(roi_filled, np.ones((region_pad * 2, region_pad * 2), np.uint8))

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    events = []
    frame_idx = 0
    history = []
    accumulated_loss = 0.0
    count = 0
    prev_area = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        yellow = create_mask(frame)
        yellow_in_region = cv2.bitwise_and(yellow, region_mask)
        area = cv2.countNonZero(yellow_in_region)

        history.append(area)
        if len(history) > smoothing:
            history.pop(0)
        smoothed = sum(history) / len(history)

        if prev_area is not None:
            delta = prev_area - smoothed  # positive = yellow decreased
            if delta > 50:  # significant decrease
                accumulated_loss += delta

            # Decay accumulated loss slowly (prevents buildup from noise)
            accumulated_loss *= 0.95

            while accumulated_loss >= ball_area:
                count += 1
                accumulated_loss -= ball_area
                events.append((frame_idx, 1, accumulated_loss))

        prev_area = smoothed
        frame_idx += 1

    cap.release()
    return events, fps, count


# ============================================================
# Run sweeps
# ============================================================
with open("samples/blue/sample1-goal.scores.json") as f:
    gt = json.load(f)

VIDEO = "samples/blue/sample1-goal.mp4"
ROI = "samples/blue/sample1-goal.roi.json"

print(f"Blue goal - Ground truth: {len(gt)} events")
print(f"{'='*60}")

# --- Approach A: Drop detection (absolute level drop) ---
print("\n--- Approach A: Yellow level drop detection ---")
for ba in [400, 500, 600, 700, 900, 1100]:
    for pad in [30, 40, 50, 60, 80]:
        for md in [100, 200, 300, 400]:
            for sm in [1, 3, 5]:
                for cd in [0, 3, 5]:
                    events, fps, count = run_disappear(
                        VIDEO, ROI, ball_area=ba, region_pad=pad,
                        min_drop=md, smoothing=sm, cooldown=cd)
                    total = sum(n for _, n, *_ in events)
                    matched, missed, fp = compare_events(events, gt)
                    if matched >= 28 and len(fp) <= 6:
                        print(f"  ba={ba:4d} pad={pad:2d} md={md:3d} sm={sm} cd={cd}: "
                              f"total={total:3d} match={matched:2d} miss={len(missed):2d} fp={len(fp):2d}")

# --- Approach B: Delta accumulation ---
print("\n--- Approach B: Delta accumulation ---")
for ba in [300, 400, 500, 600, 700, 900]:
    for pad in [30, 40, 50, 60, 80]:
        for sm in [1, 3, 5]:
            events, fps, count = run_delta_disappear(
                VIDEO, ROI, ball_area=ba, region_pad=pad, smoothing=sm)
            total = sum(n for _, n, *_ in events)
            matched, missed, fp = compare_events(events, gt)
            if matched >= 28 and len(fp) <= 6:
                print(f"  ba={ba:4d} pad={pad:2d} sm={sm}: "
                      f"total={total:3d} match={matched:2d} miss={len(missed):2d} fp={len(fp):2d}")

# Show detailed best guess
print(f"\n{'='*60}")
print("Detailed: drop detection ba=700 pad=50 md=200 sm=3 cd=0")
events, fps, count = run_disappear(VIDEO, ROI, ball_area=700, region_pad=50,
                                    min_drop=200, smoothing=3, cooldown=0)
total = sum(n for _, n, *_ in events)
print(f"Events ({len(events)}, total={total}):")
for f, n, d in events:
    print(f"  Frame {f:4d} ({f/fps:6.2f}s): +{n} (drop={d:.0f})")
matched, missed, fp = compare_events(events, gt)
print(f"Matched: {matched}/{len(gt)}  Missed: {len(missed)}  FP: {len(fp)}")
print(f"Missed: {missed}")
print(f"FP: {fp}")
