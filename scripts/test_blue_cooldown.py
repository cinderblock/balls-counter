"""Test ring area approach on blue goal WITH cooldown."""

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


def run(video_path, roi_json_path, ball_area=900, band_width=10,
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


with open("samples/blue/sample1-goal.scores.json") as f:
    gt = json.load(f)

VIDEO = "samples/blue/sample1-goal.mp4"
ROI = "samples/blue/sample1-goal.roi.json"

print(f"Blue goal - Ground truth: {len(gt)} events\n")

for ba in [400, 500, 600, 700, 900, 1100, 1400]:
    for bw in [8, 10, 12, 15]:
        for cd in [2, 3, 4, 5, 6, 8]:
            for mp in [0, 100, 200]:
                for fr in [0.4, 0.5, 0.6]:
                    events, fps = run(VIDEO, ROI, ball_area=ba, band_width=bw,
                                      cooldown=cd, min_peak=mp, fall_ratio=fr)
                    total = sum(n for _, n, *_ in events)
                    matched, missed, fp = compare_events(events, gt)
                    if matched >= 30 and len(fp) <= 5:
                        print(f"  ba={ba:4d} bw={bw:2d} cd={cd} mp={mp:3d} fr={fr}: "
                              f"total={total:3d} match={matched:2d} miss={len(missed):2d} fp={len(fp):2d}")
                    elif matched >= 28 and len(fp) <= 3:
                        print(f"  ba={ba:4d} bw={bw:2d} cd={cd} mp={mp:3d} fr={fr}: "
                              f"total={total:3d} match={matched:2d} miss={len(missed):2d} fp={len(fp):2d}")
