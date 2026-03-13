"""Test motion counter on all available samples with event-level comparison."""

import json
import sys

import cv2
import numpy as np

sys.path.insert(0, "src")
from ball_counter.detector import create_mask


def run_motion_counter(video_path, line_json_path, ball_area=900, band_width=20):
    """Run motion counter and return list of (frame, n_balls, peak_area) events."""
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
    count = 0

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

        if area > prev_area and area > 100:
            rising = True
            peak_val = max(peak_val, area)
        elif rising and area < peak_val * 0.5:
            n_balls = max(1, round(peak_val / ball_area))
            count += n_balls
            events.append((frame_idx, n_balls, peak_val))
            rising = False
            peak_val = 0

        prev_area = area
        frame_idx += 1

    cap.release()
    return events, fps, count


def compare_events(events, ground_truth, fps, tolerance_frames=15):
    """Compare detected events against ground truth frame list."""
    # Expand multi-ball events into individual frames for matching
    detected_frames = []
    for frame, n_balls, peak in events:
        for _ in range(n_balls):
            detected_frames.append(frame)

    gt = sorted(ground_truth)
    det = sorted(detected_frames)

    # Greedy matching within tolerance
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

    return matched_gt, missed, false_pos


def print_results(name, events, ground_truth, fps):
    """Print detailed comparison."""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    total = sum(n for _, n, _ in events)
    print(f"\nDetected events:")
    for frame, n_balls, peak in events:
        time = frame / fps
        print(f"  Frame {frame:4d} ({time:6.2f}s): +{n_balls} (peak={peak})")

    print(f"\nGround truth frames: {ground_truth}")
    print(f"\nTotal detected: {total}")
    print(f"Ground truth:   {len(ground_truth)}")
    print(f"Difference:     {total - len(ground_truth):+d}")

    matched, missed, false_pos = compare_events(events, ground_truth, fps)
    print(f"\nEvent matching (±15 frame tolerance):")
    print(f"  Matched:        {len(matched)}")
    print(f"  Missed GT:      {len(missed)}  frames={missed}")
    print(f"  False positives: {len(false_pos)}  frames={false_pos}")


# === Red goal outlet (drops) — has line ===
print("\n" + "#"*60)
print("# RED GOAL OUTLET (drops through bottom)")
print("#"*60)

with open("samples/red/sample1-goal-drops2.scores.json") as f:
    gt_red_drops = json.load(f)

events, fps, count = run_motion_counter(
    "samples/red/sample1-goal.mp4",
    "samples/red/sample1-goal.line.json",
    ball_area=900,
)
print_results("Red Outlet (BALL_AREA=900)", events, gt_red_drops, fps)

# Try different ball_area values
for ba in [700, 800, 1000, 1100]:
    events, fps, count = run_motion_counter(
        "samples/red/sample1-goal.mp4",
        "samples/red/sample1-goal.line.json",
        ball_area=ba,
    )
    total = sum(n for _, n, _ in events)
    matched, missed, false_pos = compare_events(events, gt_red_drops, fps)
    print(f"\n  BALL_AREA={ba}: total={total}, matched={len(matched)}, missed={len(missed)}, fp={len(false_pos)}")
