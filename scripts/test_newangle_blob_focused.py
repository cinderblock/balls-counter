"""Focused sweep around the best blob-tracking parameters to minimize FP.

The broad sweep found bw=50, min_blob=150, max_disap=5, max_dist=60 gives
68/68 matched with 8 FP. This script does a finer grid around those values.
"""

import json
import sys
import time
from itertools import product

sys.path.insert(0, "src")

import cv2
import numpy as np
from ball_counter.detector import create_mask
from ball_counter.tracker import CentroidTracker

PADDING = 150

SAMPLES = [
    {
        "name": "red",
        "video": "samples/new-angle/sample-red.mp4",
        "geom": "samples/new-angle/sample-red.geometry.json",
        "scores": "samples/new-angle/sample-red.scores.json",
        "line_indices": [0],
    },
    {
        "name": "blue",
        "video": "samples/new-angle/sample-blue.mp4",
        "geom": "samples/new-angle/sample-blue.geometry.json",
        "scores": "samples/new-angle/sample-blue.scores.json",
        "line_indices": [0, 1],
    },
]

# Focused sweep around best params
BW_VALUES = [40, 45, 50, 55, 60, 70]
MIN_BLOB_VALUES = [120, 150, 180, 200, 250]
MAX_DISAP_VALUES = [3, 4, 5, 6, 7]
MAX_DIST_VALUES = [50, 55, 60, 65, 70]
MATCH_TOLERANCE = 15


def compute_crop(line_geoms, full_w, full_h):
    all_xs, all_ys = [], []
    for lg in line_geoms:
        for p in lg["points"]:
            all_xs.append(p[0])
            all_ys.append(p[1])
    x1 = max(0, min(all_xs) - PADDING)
    y1 = max(0, min(all_ys) - PADDING)
    x2 = min(full_w, max(all_xs) + PADDING)
    y2 = min(full_h, max(all_ys) + PADDING)
    return x1, y1, x2, y2


def build_band_mask(line_geoms, ch, cw, x1, y1, bw):
    band_mask = np.zeros((ch, cw), dtype=np.uint8)
    for lg in line_geoms:
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
    return band_mask


def match_events(event_frames, gt_scores, tolerance):
    matched_gt = set()
    matched_ev = set()
    for ei, ef in enumerate(event_frames):
        for gi, gt in enumerate(gt_scores):
            if gi not in matched_gt and abs(ef - gt) <= tolerance:
                matched_gt.add(gi)
                matched_ev.add(ei)
                break
    fp = len(event_frames) - len(matched_ev)
    return len(matched_gt), fp


def run_blob_tracking(frames, band_mask, min_blob, max_disap, max_dist):
    bg_sub = cv2.createBackgroundSubtractorMOG2(
        history=60, varThreshold=50, detectShadows=False
    )
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    tracker = CentroidTracker(max_disappeared=max_disap, max_distance=max_dist)
    seen_ids = {}
    counted_ids = set()
    events = []

    for frame_idx, frame in enumerate(frames):
        fg = bg_sub.apply(frame)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, morph_kernel)
        yellow = create_mask(frame)
        moving_yellow = cv2.bitwise_and(yellow, fg)
        in_band = cv2.bitwise_and(moving_yellow, band_mask)

        n_labels, labels, stats, centroids_cc = cv2.connectedComponentsWithStats(
            in_band, connectivity=8
        )
        blob_centroids = []
        for i in range(1, n_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_blob:
                blob_centroids.append(
                    (int(centroids_cc[i, 0]), int(centroids_cc[i, 1]))
                )

        objects = tracker.update(blob_centroids)
        current_ids = set(objects.keys())
        for oid in current_ids:
            seen_ids[oid] = frame_idx

        for oid in list(seen_ids.keys()):
            if oid in counted_ids:
                continue
            if oid not in current_ids:
                counted_ids.add(oid)
                events.append(seen_ids[oid])

    return events


# Load and cache
sample_data = []
for s in SAMPLES:
    with open(s["geom"]) as f:
        geometries = json.load(f)
    with open(s["scores"]) as f:
        gt_scores = json.load(f)
    lines = [g for g in geometries if g["type"] == "line"]
    use_lines = [lines[i] for i in s["line_indices"]]

    cap = cv2.VideoCapture(s["video"])
    full_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    full_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    crop = compute_crop(use_lines, full_w, full_h)
    x1, y1, x2, y2 = crop
    cw, ch = x2 - x1, y2 - y1

    print(f"{s['name']}: caching {total_frames} frames at {cw}x{ch}...",
          end=" ", flush=True)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frames = []
    for _ in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame[y1:y2, x1:x2].copy())
    cap.release()
    print(f"{len(frames)} cached")

    sample_data.append({
        "name": s["name"], "frames": frames, "crop": crop,
        "lines": use_lines, "gt_scores": gt_scores,
    })

# Sweep
param_combos = list(product(BW_VALUES, MIN_BLOB_VALUES, MAX_DISAP_VALUES, MAX_DIST_VALUES))
total_runs = len(param_combos) * len(sample_data)
print(f"\nFocused sweep: {len(param_combos)} combos x {len(sample_data)} samples = {total_runs} runs")

results = []
t0 = time.time()
run_idx = 0

for bw, min_blob, max_disap, max_dist in param_combos:
    combo_results = {}
    for sd in sample_data:
        run_idx += 1
        x1, y1, x2, y2 = sd["crop"]
        ch, cw = y2 - y1, x2 - x1
        band_mask = build_band_mask(sd["lines"], ch, cw, x1, y1, bw)
        event_frames = run_blob_tracking(
            sd["frames"], band_mask, min_blob, max_disap, max_dist
        )
        matched, fp = match_events(event_frames, sd["gt_scores"], MATCH_TOLERANCE)
        combo_results[sd["name"]] = {
            "events": len(event_frames), "matched": matched,
            "fp": fp, "gt": len(sd["gt_scores"]),
        }

    total_matched = sum(v["matched"] for v in combo_results.values())
    total_fp = sum(v["fp"] for v in combo_results.values())
    results.append({
        "bw": bw, "min_blob": min_blob, "max_disap": max_disap, "max_dist": max_dist,
        "total_matched": total_matched, "total_fp": total_fp,
        "per_sample": combo_results,
    })

    if run_idx % 50 == 0:
        elapsed = time.time() - t0
        rate = run_idx / elapsed
        remaining = (total_runs - run_idx) / rate
        print(f"  {run_idx}/{total_runs} ({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)")

# Sort: max matched, min FP
results.sort(key=lambda r: (-r["total_matched"], r["total_fp"]))

# Show only perfect-match results (68/68)
perfect = [r for r in results if r["total_matched"] == 68]
print(f"\nPerfect match results (68/68): {len(perfect)} combos")
print(f"{'bw':>3} {'mblob':>5} {'mdis':>4} {'mdist':>5}  "
      f"{'red_m':>5} {'red_fp':>6} {'blu_m':>5} {'blu_fp':>6}  "
      f"{'total_fp':>8}")
print("-" * 65)

for r in perfect[:40]:
    red = r["per_sample"]["red"]
    blue = r["per_sample"]["blue"]
    print(f"{r['bw']:3d} {r['min_blob']:5d} {r['max_disap']:4d} {r['max_dist']:5d}  "
          f"{red['matched']:2d}/{red['gt']:2d} {red['fp']:6d} "
          f"{blue['matched']:2d}/{blue['gt']:2d} {blue['fp']:6d}  "
          f"{r['total_fp']:8d}")

if not perfect:
    print("  (none — showing top results instead)")
    for r in results[:20]:
        red = r["per_sample"]["red"]
        blue = r["per_sample"]["blue"]
        print(f"{r['bw']:3d} {r['min_blob']:5d} {r['max_disap']:4d} {r['max_dist']:5d}  "
              f"{red['matched']:2d}/{red['gt']:2d} {red['fp']:6d} "
              f"{blue['matched']:2d}/{blue['gt']:2d} {blue['fp']:6d}  "
              f"{r['total_fp']:8d}")

elapsed = time.time() - t0
print(f"\nDone ({elapsed:.0f}s)")
