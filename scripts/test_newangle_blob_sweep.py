"""Parameter sweep: blob-tracking approach for new-angle samples.

Instead of peak detection on aggregate area, this tracks individual blobs
(connected components of moving yellow pixels) through the counting zone.
Each unique blob that appears and then disappears = 1 ball scored.

Runs both red and blue samples in one sweep.
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
        "line_indices": [0],  # line-1 only (red/left goal)
    },
    {
        "name": "blue",
        "video": "samples/new-angle/sample-blue.mp4",
        "geom": "samples/new-angle/sample-blue.geometry.json",
        "scores": "samples/new-angle/sample-blue.scores.json",
        "line_indices": [0, 1],  # combined lines
    },
]

# Sweep parameters
BW_VALUES = [15, 25, 35, 50]         # band_width: wider than before
MIN_BLOB_VALUES = [30, 80, 150]       # min connected component area
MAX_DISAP_VALUES = [2, 5, 10]         # max_disappeared for tracker
MAX_DIST_VALUES = [40, 60, 100]       # max_distance for tracker matching
MATCH_TOLERANCE = 15                  # frames tolerance for GT matching


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
    """Match detected event frames to ground truth scores."""
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
    """Run blob tracking on pre-cached frames. Returns list of event frame indices."""
    bg_sub = cv2.createBackgroundSubtractorMOG2(
        history=60, varThreshold=50, detectShadows=False
    )
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    tracker = CentroidTracker(max_disappeared=max_disap, max_distance=max_dist)

    # Track which IDs we've seen in the zone and when they were last seen
    seen_ids = {}       # id -> last_frame_seen_in_zone
    counted_ids = set()  # IDs already counted
    events = []         # (frame_idx,) for each scored ball

    for frame_idx, frame in enumerate(frames):
        fg = bg_sub.apply(frame)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, morph_kernel)
        yellow = create_mask(frame)
        moving_yellow = cv2.bitwise_and(yellow, fg)
        in_band = cv2.bitwise_and(moving_yellow, band_mask)

        # Find connected components
        n_labels, labels, stats, centroids_cc = cv2.connectedComponentsWithStats(
            in_band, connectivity=8
        )

        # Filter blobs by minimum area, collect centroids
        blob_centroids = []
        for i in range(1, n_labels):  # skip background
            if stats[i, cv2.CC_STAT_AREA] >= min_blob:
                cx = centroids_cc[i, 0]
                cy = centroids_cc[i, 1]
                blob_centroids.append((int(cx), int(cy)))

        # Update tracker
        objects = tracker.update(blob_centroids)

        # Record which IDs are currently in the zone
        current_ids = set(objects.keys())
        for oid in current_ids:
            seen_ids[oid] = frame_idx

        # Check for IDs that have disappeared from the zone
        # (they were in seen_ids but are no longer tracked, or have been
        # disappeared for a while)
        for oid in list(seen_ids.keys()):
            if oid in counted_ids:
                continue
            if oid not in current_ids:
                # This blob has left the zone — count it as a score
                counted_ids.add(oid)
                events.append(seen_ids[oid])  # use last frame it was seen

    return events


# Load and cache all samples
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
    fps = cap.get(cv2.CAP_PROP_FPS)

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
        "name": s["name"],
        "frames": frames,
        "crop": crop,
        "lines": use_lines,
        "gt_scores": gt_scores,
        "full_w": full_w,
        "full_h": full_h,
    })


# Run sweep
param_combos = list(product(BW_VALUES, MIN_BLOB_VALUES, MAX_DISAP_VALUES, MAX_DIST_VALUES))
total_runs = len(param_combos) * len(sample_data)
print(f"\nSweeping {len(param_combos)} param combos x {len(sample_data)} samples = {total_runs} runs")

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
            "events": len(event_frames),
            "matched": matched,
            "fp": fp,
            "gt": len(sd["gt_scores"]),
        }

    # Combined score
    total_matched = sum(v["matched"] for v in combo_results.values())
    total_gt = sum(v["gt"] for v in combo_results.values())
    total_fp = sum(v["fp"] for v in combo_results.values())
    total_events = sum(v["events"] for v in combo_results.values())

    results.append({
        "bw": bw, "min_blob": min_blob, "max_disap": max_disap, "max_dist": max_dist,
        "total_matched": total_matched, "total_gt": total_gt,
        "total_fp": total_fp, "total_events": total_events,
        "per_sample": combo_results,
    })

    if run_idx % 20 == 0:
        elapsed = time.time() - t0
        rate = run_idx / elapsed
        remaining = (total_runs - run_idx) / rate
        print(f"  {run_idx}/{total_runs} ({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)")

# Sort by total matched (desc), then FP (asc)
results.sort(key=lambda r: (-r["total_matched"], r["total_fp"]))

elapsed = time.time() - t0
print(f"\nTop results ({elapsed:.0f}s total):")
print(f"{'bw':>3} {'mblob':>5} {'mdis':>4} {'mdist':>5}  "
      f"{'red_m':>5} {'red_fp':>6} {'blu_m':>5} {'blu_fp':>6}  "
      f"{'total_m':>7} {'total_fp':>8}")
print("-" * 75)

for r in results[:30]:
    red = r["per_sample"].get("red", {})
    blue = r["per_sample"].get("blue", {})
    marker = ""
    if r["total_matched"] >= 55:
        marker = " ***"
    elif r["total_matched"] >= 45:
        marker = " **"
    elif r["total_matched"] >= 35:
        marker = " *"
    print(f"{r['bw']:3d} {r['min_blob']:5d} {r['max_disap']:4d} {r['max_dist']:5d}  "
          f"{red.get('matched', 0):2d}/{red.get('gt', 0):2d} {red.get('fp', 0):6d} "
          f"{blue.get('matched', 0):2d}/{blue.get('gt', 0):2d} {blue.get('fp', 0):6d}  "
          f"{r['total_matched']:2d}/{r['total_gt']:2d} {r['total_fp']:8d}{marker}")

print("\nDone")
