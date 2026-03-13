"""Test multiple counting approaches on the blue goal ROI."""

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
    ring = cv2.subtract(outer_dilated, inner_eroded)
    return ring


def load_video_and_masks(roi_json_path, video_path, band_width=10):
    with open(roi_json_path) as f:
        roi_data = json.load(f)
    roi_pts = np.array(roi_data["roi"], dtype=np.int32)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    ret, first = cap.read()
    h, w = first.shape[:2]

    ring_mask = make_ring_mask(h, w, roi_pts, band_width)

    # Compute centroid of ROI for direction filtering
    M = cv2.moments(roi_pts.reshape(-1, 1, 2))
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return cap, fps, ring_mask, (cx, cy), roi_pts


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


def print_results(name, events, gt, fps):
    total = sum(n for _, n, *_ in events)
    matched, missed, fp = compare_events(events, gt)
    print(f"\n  {name}")
    print(f"  Total: {total}  Matched: {matched}/{len(gt)}  Missed: {len(missed)}  FP: {len(fp)}")
    if len(missed) <= 5:
        print(f"  Missed frames: {missed}")
    if len(fp) <= 10:
        print(f"  FP frames: {fp}")
    return matched, len(missed), len(fp)


# ============================================================
# Approach 1: Blob counting (connected components, not area)
# Each peak in blob count = scoring event, peak value = n_balls
# ============================================================
def approach_blob_count(video_path, roi_json_path, band_width=10,
                        min_blobs=1, cooldown=0, min_blob_area=50):
    cap, fps, ring_mask, centroid, roi_pts = load_video_and_masks(
        roi_json_path, video_path, band_width)

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

        # Count blobs instead of total area
        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            moving_in_ring, connectivity=8)
        # Filter small blobs (noise)
        blob_count = 0
        for i in range(1, n_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_blob_area:
                blob_count += 1

        if cooldown_remaining > 0:
            cooldown_remaining -= 1
            prev_blobs = blob_count
            frame_idx += 1
            continue

        if blob_count > prev_blobs and blob_count >= min_blobs:
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


# ============================================================
# Approach 2: Area-based with cooldown (debounce after scoring)
# ============================================================
def approach_area_cooldown(video_path, roi_json_path, ball_area=1100,
                           band_width=10, cooldown=5, min_peak=0):
    cap, fps, ring_mask, centroid, roi_pts = load_video_and_masks(
        roi_json_path, video_path, band_width)

    bg_sub = cv2.createBackgroundSubtractorMOG2(
        history=60, varThreshold=50, detectShadows=False)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    events = []
    frame_idx = 0
    prev_area = 0
    rising = False
    peak_val = 0
    cooldown_remaining = 0
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

        if cooldown_remaining > 0:
            cooldown_remaining -= 1
            prev_area = area
            frame_idx += 1
            continue

        if area > prev_area and area > 100:
            rising = True
            peak_val = max(peak_val, area)
        elif rising and area < peak_val * 0.5:
            if peak_val >= min_peak:
                n_balls = max(1, round(peak_val / ball_area))
                count += n_balls
                events.append((frame_idx, n_balls, peak_val))
                cooldown_remaining = cooldown
            rising = False
            peak_val = 0

        prev_area = area
        frame_idx += 1

    cap.release()
    return events, fps


# ============================================================
# Approach 3: Direction-filtered blobs (only inward motion)
# Track blob centroids frame-to-frame, only count blobs moving
# toward the ROI centroid.
# ============================================================
def approach_direction_filtered(video_path, roi_json_path, band_width=10,
                                min_blob_area=50, cooldown=0):
    cap, fps, ring_mask, roi_centroid, roi_pts = load_video_and_masks(
        roi_json_path, video_path, band_width)

    bg_sub = cv2.createBackgroundSubtractorMOG2(
        history=60, varThreshold=50, detectShadows=False)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    events = []
    frame_idx = 0
    prev_centroids = []
    cooldown_remaining = 0
    inward_count = 0
    prev_inward = 0
    rising = False
    peak_inward = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        fg_mask = bg_sub.apply(frame)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        yellow = create_mask(frame)
        moving_yellow = cv2.bitwise_and(yellow, fg_mask)
        moving_in_ring = cv2.bitwise_and(moving_yellow, ring_mask)

        n_labels, labels, stats, centroids_arr = cv2.connectedComponentsWithStats(
            moving_in_ring, connectivity=8)

        curr_centroids = []
        for i in range(1, n_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_blob_area:
                cx, cy = centroids_arr[i]
                curr_centroids.append((cx, cy))

        # Match current blobs to previous by proximity
        inward_blobs = 0
        for cx, cy in curr_centroids:
            # Check if this blob is closer to ROI centroid than any previous blob
            dist_to_center = ((cx - roi_centroid[0])**2 + (cy - roi_centroid[1])**2)**0.5
            best_prev_dist = float('inf')
            for px, py in prev_centroids:
                d = ((cx - px)**2 + (cy - py)**2)**0.5
                if d < 30:  # same blob
                    prev_dist = ((px - roi_centroid[0])**2 + (py - roi_centroid[1])**2)**0.5
                    if prev_dist < best_prev_dist:
                        best_prev_dist = prev_dist
                    if dist_to_center < prev_dist:
                        inward_blobs += 1
                    break

        if cooldown_remaining > 0:
            cooldown_remaining -= 1
        else:
            if inward_blobs > prev_inward and inward_blobs >= 1:
                rising = True
                peak_inward = max(peak_inward, inward_blobs)
            elif rising and inward_blobs < max(1, peak_inward * 0.5):
                events.append((frame_idx, peak_inward))
                cooldown_remaining = cooldown
                rising = False
                peak_inward = 0

        prev_inward = inward_blobs
        prev_centroids = curr_centroids
        frame_idx += 1

    cap.release()
    return events, fps


# ============================================================
# Approach 4: New-blob entrance detector
# Count blobs that appear in the ring without a match in prev frame
# (new arrivals). Peak detection on new arrivals.
# ============================================================
def approach_new_blobs(video_path, roi_json_path, band_width=10,
                       min_blob_area=50, match_dist=30, cooldown=0):
    cap, fps, ring_mask, centroid, roi_pts = load_video_and_masks(
        roi_json_path, video_path, band_width)

    bg_sub = cv2.createBackgroundSubtractorMOG2(
        history=60, varThreshold=50, detectShadows=False)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    events = []
    frame_idx = 0
    prev_centroids = []
    new_count_signal = 0
    prev_signal = 0
    rising = False
    peak_signal = 0
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

        n_labels, labels, stats, centroids_arr = cv2.connectedComponentsWithStats(
            moving_in_ring, connectivity=8)

        curr_centroids = []
        for i in range(1, n_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_blob_area:
                cx, cy = centroids_arr[i]
                curr_centroids.append((cx, cy))

        # Count blobs with no match in previous frame
        new_blobs = 0
        for cx, cy in curr_centroids:
            matched = False
            for px, py in prev_centroids:
                if ((cx - px)**2 + (cy - py)**2)**0.5 < match_dist:
                    matched = True
                    break
            if not matched:
                new_blobs += 1

        if cooldown_remaining > 0:
            cooldown_remaining -= 1
        else:
            if new_blobs > prev_signal and new_blobs >= 1:
                rising = True
                peak_signal = max(peak_signal, new_blobs)
            elif rising and new_blobs < max(1, peak_signal * 0.5):
                events.append((frame_idx, peak_signal))
                cooldown_remaining = cooldown
                rising = False
                peak_signal = 0

        prev_signal = new_blobs
        prev_centroids = curr_centroids
        frame_idx += 1

    cap.release()
    return events, fps


# ============================================================
# Run all approaches
# ============================================================
with open("samples/blue/sample1-goal.scores.json") as f:
    gt = json.load(f)

VIDEO = "samples/blue/sample1-goal.mp4"
ROI = "samples/blue/sample1-goal.roi.json"

print(f"Blue goal - Ground truth: {len(gt)} events")
print(f"{'='*60}")

# --- Approach 1: Blob count ---
print("\n--- Approach 1: Blob counting (peak on blob count) ---")
for min_blob in [30, 50, 80]:
    for cd in [0, 3, 5, 8]:
        for bw in [8, 10, 12, 15]:
            events, fps = approach_blob_count(VIDEO, ROI, band_width=bw,
                                              cooldown=cd, min_blob_area=min_blob)
            total = sum(n for _, n, *_ in events)
            matched, missed, fp = compare_events(events, gt)
            if matched >= 28 and len(fp) <= 8:
                print(f"  bw={bw:2d} mb={min_blob:2d} cd={cd}: "
                      f"total={total:3d} match={matched:2d} miss={len(missed):2d} fp={len(fp):2d}")

# --- Approach 2: Area with cooldown ---
print("\n--- Approach 2: Area-based with cooldown ---")
for ba in [900, 1100, 1400]:
    for cd in [3, 5, 8, 10]:
        for mp in [0, 200, 400]:
            events, fps = approach_area_cooldown(VIDEO, ROI, ball_area=ba,
                                                 cooldown=cd, min_peak=mp)
            total = sum(n for _, n, *_ in events)
            matched, missed, fp = compare_events(events, gt)
            if matched >= 28 and len(fp) <= 8:
                print(f"  ba={ba:4d} cd={cd:2d} mp={mp:3d}: "
                      f"total={total:3d} match={matched:2d} miss={len(missed):2d} fp={len(fp):2d}")

# --- Approach 3: Direction filtered ---
print("\n--- Approach 3: Direction-filtered blobs ---")
for mb in [30, 50]:
    for cd in [0, 3, 5]:
        for bw in [8, 10, 15]:
            events, fps = approach_direction_filtered(VIDEO, ROI, band_width=bw,
                                                      min_blob_area=mb, cooldown=cd)
            total = sum(n for _, n, *_ in events)
            matched, missed, fp = compare_events(events, gt)
            if matched >= 20 or (matched >= 15 and len(fp) <= 5):
                print(f"  bw={bw:2d} mb={mb:2d} cd={cd}: "
                      f"total={total:3d} match={matched:2d} miss={len(missed):2d} fp={len(fp):2d}")

# --- Approach 4: New blob entrance ---
print("\n--- Approach 4: New blob entrance detector ---")
for mb in [30, 50, 80]:
    for md in [20, 30, 40]:
        for cd in [0, 3, 5]:
            for bw in [8, 10, 15]:
                events, fps = approach_new_blobs(VIDEO, ROI, band_width=bw,
                                                 min_blob_area=mb, match_dist=md,
                                                 cooldown=cd)
                total = sum(n for _, n, *_ in events)
                matched, missed, fp = compare_events(events, gt)
                if matched >= 25 and len(fp) <= 8:
                    print(f"  bw={bw:2d} mb={mb:2d} md={md:2d} cd={cd}: "
                          f"total={total:3d} match={matched:2d} miss={len(missed):2d} fp={len(fp):2d}")
