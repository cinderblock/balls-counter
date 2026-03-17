"""Parameter sweep for new-angle blue sample — windowed + pre-cached."""

import json
import sys
import time

sys.path.insert(0, "src")

import cv2
import numpy as np
from ball_counter.detector import create_mask

VIDEO = "samples/new-angle/sample-blue.mp4"
GEOM_FILE = "samples/new-angle/sample-blue.geometry.json"
SCORES_FILE = "samples/new-angle/sample-blue.scores.json"
PADDING = 150

with open(GEOM_FILE) as f:
    geometries = json.load(f)
with open(SCORES_FILE) as f:
    gt_scores = json.load(f)

lines = [g for g in geometries if g["type"] == "line"]
print(f"Ground truth: {len(gt_scores)} scores, Lines: {len(lines)}")

# Blue lines form one continuous path — always test as combined
test_configs = [("combined", lines)]

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

cap = cv2.VideoCapture(VIDEO)
full_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
full_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Video: {full_w}x{full_h} @ {fps}fps, {total_frames} frames")

config_caches = {}
for name, line_geoms in test_configs:
    crop = compute_crop(line_geoms, full_w, full_h)
    x1, y1, x2, y2 = crop
    cw, ch = x2 - x1, y2 - y1

    cache_key = crop
    if cache_key in config_caches:
        config_caches[name] = config_caches[cache_key]
        print(f"  {name}: reusing cache for crop ({x1},{y1})-({x2},{y2})")
        continue

    print(f"  {name}: caching {total_frames} frames at {cw}x{ch} "
          f"({cw*ch/(full_w*full_h)*100:.1f}% of full)...", end=" ", flush=True)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frames = []
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame[y1:y2, x1:x2].copy())

    config_caches[name] = {"frames": frames, "crop": crop}
    config_caches[cache_key] = config_caches[name]
    print(f"{len(frames)} frames cached")

cap.release()

BA_VALUES = [300, 500, 700, 900, 1200, 1500, 2000]
BW_VALUES = [10, 15, 20, 25]
FR_VALUES = [0.5, 0.6, 0.7]

morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
results = []

t0 = time.time()
total_combos = len(test_configs) * len(BA_VALUES) * len(BW_VALUES) * len(FR_VALUES)
combo_idx = 0

for config_name, line_geoms in test_configs:
    cache = config_caches[config_name]
    frames = cache["frames"]
    crop = cache["crop"]
    x1, y1, x2, y2 = crop
    ch, cw = y2 - y1, x2 - x1

    for ba in BA_VALUES:
        for bw in BW_VALUES:
            for fr in FR_VALUES:
                combo_idx += 1

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

                bg_sub = cv2.createBackgroundSubtractorMOG2(
                    history=60, varThreshold=50, detectShadows=False
                )

                prev_area = 0
                rising = False
                peak_val = 0
                events = []

                for frame_idx, frame in enumerate(frames):
                    fg = bg_sub.apply(frame)
                    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, morph_kernel)
                    yellow = create_mask(frame)
                    moving_yellow = cv2.bitwise_and(yellow, fg)
                    in_band = cv2.bitwise_and(moving_yellow, band_mask)
                    area = cv2.countNonZero(in_band)

                    if area > prev_area and area > 100:
                        rising = True
                        peak_val = max(peak_val, area)
                    elif rising and area < peak_val * fr:
                        if peak_val >= 100:
                            n = max(1, round(peak_val / ba))
                            events.append((frame_idx, n, peak_val))
                        rising = False
                        peak_val = 0

                    prev_area = area

                total_count = sum(e[1] for e in events)

                matched_gt = set()
                matched_ev = set()
                for ei, ev in enumerate(events):
                    for gi, gt in enumerate(gt_scores):
                        if gi not in matched_gt and abs(ev[0] - gt) <= 15:
                            matched_gt.add(gi)
                            matched_ev.add(ei)
                            break

                fp = len(events) - len(matched_ev)

                results.append({
                    "line": config_name, "ba": ba, "bw": bw, "fr": fr,
                    "events": len(events), "count": total_count,
                    "matched": len(matched_gt), "fp": fp,
                })

    elapsed = time.time() - t0
    rate = combo_idx / elapsed
    remaining = (total_combos - combo_idx) / rate if rate > 0 else 0
    print(f"  {config_name} done ({combo_idx}/{total_combos}, {elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)")

results.sort(key=lambda r: (-r["matched"], r["fp"]))

print(f"\nTop results ({time.time() - t0:.0f}s total):")
for r in results[:40]:
    marker = ""
    if r["matched"] >= 30 and r["fp"] <= 5:
        marker = " **"
    elif r["matched"] >= 25:
        marker = " *"
    print(f"  {r['line']:10s} ba={r['ba']:5d} bw={r['bw']:2d} fr={r['fr']:.1f}: "
          f"count={r['count']:3d} events={r['events']:3d} "
          f"matched={r['matched']:2d}/{len(gt_scores)} FP={r['fp']:2d}{marker}")

print("\nDone")
