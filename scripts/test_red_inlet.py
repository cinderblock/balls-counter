"""Test all approaches on red goal inlet ROI."""

import json
import sys

import cv2
import numpy as np

sys.path.insert(0, "src")
from ball_counter.detector import create_mask


def make_ring_mask(h, w, roi_pts, band_width=10):
    outer = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(outer, [roi_pts], 255)
    outer_dilated = cv2.dilate(outer, np.ones((band_width, band_width), np.uint8))
    inner_eroded = cv2.erode(outer, np.ones((band_width, band_width), np.uint8))
    return cv2.subtract(outer_dilated, inner_eroded)


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


def run_ring_area(video_path, roi_json_path, ball_area=900, band_width=10,
                  min_peak=0, fall_ratio=0.5, cooldown=0):
    with open(roi_json_path) as f:
        roi_pts = np.array(json.load(f)["roi"], dtype=np.int32)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    ret, first = cap.read()
    h, w = first.shape[:2]
    ring_mask = make_ring_mask(h, w, roi_pts, band_width)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    bg_sub = cv2.createBackgroundSubtractorMOG2(
        history=60, varThreshold=50, detectShadows=False)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    events = []
    frame_idx = 0
    prev_area = 0
    rising = False
    peak_val = 0
    cooldown_remaining = 0

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

        if cooldown_remaining > 0:
            cooldown_remaining -= 1
            prev_area = area
            frame_idx += 1
            continue

        if area > prev_area and area > 100:
            rising = True
            peak_val = max(peak_val, area)
        elif rising and area < peak_val * fall_ratio:
            if peak_val >= min_peak:
                n_balls = max(1, round(peak_val / ball_area))
                events.append((frame_idx, n_balls, peak_val))
                cooldown_remaining = cooldown
            rising = False
            peak_val = 0

        prev_area = area
        frame_idx += 1

    cap.release()
    return events, fps


def run_blob_count(video_path, roi_json_path, band_width=10,
                   min_blob_area=50, cooldown=0):
    with open(roi_json_path) as f:
        roi_pts = np.array(json.load(f)["roi"], dtype=np.int32)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    ret, first = cap.read()
    h, w = first.shape[:2]
    ring_mask = make_ring_mask(h, w, roi_pts, band_width)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    bg_sub = cv2.createBackgroundSubtractorMOG2(
        history=60, varThreshold=50, detectShadows=False)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    events = []
    frame_idx = 0
    prev_blobs = 0
    rising = False
    peak_blobs = 0
    cooldown_remaining = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        fg_mask = bg_sub.apply(frame)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        yellow = create_mask(frame)
        moving_yellow = cv2.bitwise_and(yellow, fg_mask)
        moving_in_ring = cv2.bitwise_and(moving_yellow, ring_mask)

        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            moving_in_ring, connectivity=8)
        blob_count = sum(1 for i in range(1, n_labels)
                         if stats[i, cv2.CC_STAT_AREA] >= min_blob_area)

        if cooldown_remaining > 0:
            cooldown_remaining -= 1
            prev_blobs = blob_count
            frame_idx += 1
            continue

        if blob_count > prev_blobs and blob_count >= 1:
            rising = True
            peak_blobs = max(peak_blobs, blob_count)
        elif rising and blob_count < peak_blobs * 0.5:
            events.append((frame_idx, peak_blobs))
            cooldown_remaining = cooldown
            rising = False
            peak_blobs = 0

        prev_blobs = blob_count
        frame_idx += 1

    cap.release()
    return events, fps


VIDEO = "samples/red/sample1-goal.mp4"
ROI = "samples/red/sample1-goal-inlet.roi.json"

with open("samples/red/sample1-goal-inlet.scores.json") as f:
    gt = json.load(f)

print(f"Red inlet - Ground truth: {len(gt)} events")
print(f"GT frames: {gt}")
print(f"{'='*60}")

# Ring area approach
print("\n--- Ring area approach ---")
for ba in [500, 700, 900, 1100]:
    for bw in [8, 10, 12, 15]:
        for cd in [0, 3, 5, 8]:
            for mp in [0, 200, 400]:
                events, fps = run_ring_area(VIDEO, ROI, ball_area=ba,
                                            band_width=bw, cooldown=cd, min_peak=mp)
                total = sum(n for _, n, *_ in events)
                matched, missed, fp = compare_events(events, gt)
                if matched >= 15 and len(fp) <= 5:
                    print(f"  ba={ba:4d} bw={bw:2d} cd={cd} mp={mp:3d}: "
                          f"total={total:3d} match={matched:2d} miss={len(missed):2d} fp={len(fp):2d}")

# Blob count approach
print("\n--- Blob count approach ---")
for mb in [30, 50, 80]:
    for bw in [8, 10, 12, 15]:
        for cd in [0, 3, 5, 8]:
            events, fps = run_blob_count(VIDEO, ROI, band_width=bw,
                                         min_blob_area=mb, cooldown=cd)
            total = sum(n for _, n, *_ in events)
            matched, missed, fp = compare_events(events, gt)
            if matched >= 15 and len(fp) <= 5:
                print(f"  bw={bw:2d} mb={mb:2d} cd={cd}: "
                      f"total={total:3d} match={matched:2d} miss={len(missed):2d} fp={len(fp):2d}")

# Also show a detailed default run
print(f"\n{'='*60}")
print("Detailed: ring area ba=900 bw=10 cd=0 mp=0")
events, fps = run_ring_area(VIDEO, ROI)
total = sum(n for _, n, *_ in events)
print(f"Events ({len(events)}, total={total}):")
for f, n, p in events:
    print(f"  Frame {f:4d} ({f/fps:6.2f}s): +{n} (peak={p})")
matched, missed, fp = compare_events(events, gt)
print(f"Matched: {matched}/{len(gt)}  Missed: {len(missed)}  FP: {len(fp)}")
print(f"Missed: {missed}")
print(f"FP: {fp}")
