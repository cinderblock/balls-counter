"""Tune motion counter parameters for best event-level accuracy."""

import json
import sys

import cv2
import numpy as np

sys.path.insert(0, "src")
from ball_counter.detector import create_mask


def run_motion_counter(video_path, line_json_path, ball_area=900, band_width=20,
                       min_peak=0, fall_ratio=0.5, min_rise=100):
    with open(line_json_path) as f:
        line_data = json.load(f)
    p1, p2 = line_data["line"]

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    ret, first = cap.read()
    h, w = first.shape[:2]

    band_mask = np.zeros((h, w), dtype=np.uint8)
    band_pts = np.array([
        [p1[0], p1[1] - band_width], [p2[0], p2[1] - band_width],
        [p2[0], p2[1] + band_width], [p1[0], p1[1] + band_width]
    ], dtype=np.int32)
    cv2.fillPoly(band_mask, [band_pts], 255)
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

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        fg_mask = bg_sub.apply(frame)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        yellow = create_mask(frame)
        moving_yellow = cv2.bitwise_and(yellow, fg_mask)
        moving_in_band = cv2.bitwise_and(moving_yellow, band_mask)
        area = cv2.countNonZero(moving_in_band)

        if area > prev_area and area > min_rise:
            rising = True
            peak_val = max(peak_val, area)
        elif rising and area < peak_val * fall_ratio:
            if peak_val >= min_peak:
                n_balls = max(1, round(peak_val / ball_area))
                events.append((frame_idx, n_balls, peak_val))
            rising = False
            peak_val = 0

        prev_area = area
        frame_idx += 1

    cap.release()
    return events, fps


def compare_events(events, ground_truth, fps, tolerance_frames=15):
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


with open("samples/red/sample1-goal-drops2.scores.json") as f:
    gt = json.load(f)

print("Tuning motion counter on red goal outlet")
print(f"Ground truth: {len(gt)} events\n")

best_score = 0
best_params = None

for ball_area in [700, 800, 900, 1000]:
    for band_width in [15, 20, 25, 30]:
        for min_peak in [0, 200, 300, 400, 500]:
            for fall_ratio in [0.4, 0.5, 0.6]:
                events, fps = run_motion_counter(
                    "samples/red/sample1-goal.mp4",
                    "samples/red/sample1-goal.line.json",
                    ball_area=ball_area,
                    band_width=band_width,
                    min_peak=min_peak,
                    fall_ratio=fall_ratio,
                )
                total = sum(n for _, n, _ in events)
                matched, missed, fp = compare_events(events, gt, fps)
                score = matched - len(fp) * 0.5 - len(missed) * 0.5

                if score > best_score:
                    best_score = score
                    best_params = {
                        "ball_area": ball_area,
                        "band_width": band_width,
                        "min_peak": min_peak,
                        "fall_ratio": fall_ratio,
                        "total": total,
                        "matched": matched,
                        "missed": len(missed),
                        "fp": len(fp),
                        "missed_frames": missed,
                        "fp_frames": fp,
                    }

                if matched >= 14 and len(fp) <= 1:
                    print(f"  ba={ball_area} bw={band_width} mp={min_peak} fr={fall_ratio}: "
                          f"total={total} match={matched} miss={len(missed)} fp={len(fp)}")

print(f"\n{'='*60}")
print(f"Best params (score={best_score:.1f}):")
for k, v in best_params.items():
    print(f"  {k}: {v}")
