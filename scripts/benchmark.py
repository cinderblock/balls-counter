#!/usr/bin/env python3
"""Benchmark the MotionCounter detector against human-annotated clips.

Usage:
    uv run python scripts/benchmark.py configs/clips --config configs/live.json
    uv run python scripts/benchmark.py configs/clips --config configs/live.json --goal red-goal -v
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ball_counter.config import load_configs
from ball_counter.counter import MotionCounter, MotionEvent


# ── Clip loading ──────────────────────────────────────────────────────────────

def load_annotated_clip(json_path: Path) -> dict | None:
    """Load a clip sidecar. Returns None if it has no human annotations."""
    d = json.loads(json_path.read_text())
    if not d.get("annotations"):
        return None
    return d


def get_all_marks(clip: dict) -> list[float]:
    """Aggregate human marks from all reviewers, deduplicating within 0.5s."""
    times = []
    for ann in clip["annotations"].values():
        for m in ann.get("marks", []):
            times.append(float(m["video_time"]))
    times.sort()
    deduped: list[float] = []
    for t in times:
        if not deduped or t - deduped[-1] > 0.5:
            deduped.append(t)
    return deduped


# ── Detector replay ───────────────────────────────────────────────────────────

def build_motion_counter(goal_config, crop_w: int, crop_h: int, **param_overrides) -> MotionCounter:
    """Build a MotionCounter for a pre-cropped frame of size (crop_h, crop_w).

    The clip MP4 stores raw frames at frame[y1:y2, x1:x2] (before downsampling).
    We translate the line/roi from full-frame coords to crop-local coords here.
    """
    x1, y1 = 0, 0
    if goal_config.crop_override:
        x1, y1 = goal_config.crop_override[0], goal_config.crop_override[1]

    ds = goal_config.downsample
    scaled_w = max(1, int(crop_w * ds))
    scaled_h = max(1, int(crop_h * ds))

    def offset_scale(pts: list) -> list:
        return [[int((p[0] - x1) * ds), int((p[1] - y1) * ds)] for p in pts]

    line = tuple(offset_scale(goal_config.line)) if goal_config.line else None
    roi = offset_scale(goal_config.roi_points) if goal_config.roi_points else None

    params = dict(
        ball_area=goal_config.ball_area,
        band_width=goal_config.band_width,
        min_peak=goal_config.min_peak,
        fall_ratio=goal_config.fall_ratio,
        cooldown=goal_config.cooldown,
        hsv_low=goal_config.hsv_low,
        hsv_high=goal_config.hsv_high,
    )
    params.update(param_overrides)

    return MotionCounter(
        frame_shape=(scaled_h, scaled_w),
        line=line,
        roi=roi,
        **params,
    )


def run_detector(mp4_path: Path, goal_config, fps_hint: float = 30.0,
                 **param_overrides) -> tuple[list[float], float]:
    """Run MotionCounter on an MP4 clip. Returns (event_times_sec, fps)."""
    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        return [], fps_hint

    actual_fps = cap.get(cv2.CAP_PROP_FPS) or fps_hint
    ds = goal_config.downsample

    ret, first = cap.read()
    if not ret:
        cap.release()
        return [], actual_fps

    h, w = first.shape[:2]
    counter = build_motion_counter(goal_config, w, h, **param_overrides)

    def feed(frame: np.ndarray) -> MotionEvent | None:
        if ds != 1.0:
            sw, sh = max(1, int(w * ds)), max(1, int(h * ds))
            frame = cv2.resize(frame, (sw, sh))
        return counter.process_frame(frame)

    feed(first)  # initialize background model
    event_times: list[float] = []
    frame_idx = 1

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        ev = feed(frame)
        if ev is not None:
            event_times.append(frame_idx / actual_fps)
        frame_idx += 1

    cap.release()
    return event_times, actual_fps


# ── Matching & metrics ────────────────────────────────────────────────────────

def match_events(auto: list[float], human: list[float],
                 tolerance: float = 1.0) -> tuple[int, int, int]:
    """Greedy nearest-neighbour matching. Returns (TP, FP, FN)."""
    pairs = sorted(
        [(abs(a - h), i, j) for i, a in enumerate(auto) for j, h in enumerate(human)
         if abs(a - h) < tolerance]
    )
    used_a: set[int] = set()
    used_h: set[int] = set()
    for _, i, j in pairs:
        if i not in used_a and j not in used_h:
            used_a.add(i)
            used_h.add(j)
    tp = len(used_a)
    return tp, len(auto) - tp, len(human) - tp


# ── Main benchmark ────────────────────────────────────────────────────────────

def benchmark(clips_dir: Path, config_path: Path | None, goal_filter: str | None,
              tolerance: float, verbose: bool, param_overrides: dict) -> dict:
    goal_configs: dict = {}
    if config_path:
        sources, _pfms = load_configs(config_path)
        for src in sources:
            for g in src.goals:
                goal_configs[g.name] = g

    clips = []
    for jp in sorted(clips_dir.glob("*.json")):
        if jp.stem == "reviewers":
            continue
        mp4 = jp.with_suffix(".mp4")
        if not mp4.exists():
            continue
        clip = load_annotated_clip(jp)
        if clip is None:
            continue
        if goal_filter and clip.get("goal") != goal_filter:
            continue
        clips.append((jp, mp4, clip))

    if not clips:
        print(f"No annotated clips found in {clips_dir}", file=sys.stderr)
        return {}

    if verbose:
        print(f"Evaluating {len(clips)} annotated clip(s)  tolerance={tolerance}s\n")

    total_tp = total_fp = total_fn = 0
    total_frames = 0
    total_wall = 0.0

    for jp, mp4, clip in clips:
        goal_name = clip.get("goal", "")
        gc = goal_configs.get(goal_name)
        if gc is None:
            if verbose:
                print(f"  SKIP {jp.stem}: no config for goal '{goal_name}'")
            continue

        human = get_all_marks(clip)
        if not human:
            continue

        t0 = time.perf_counter()
        auto, fps = run_detector(mp4, gc, clip.get("fps", 30.0), **param_overrides)
        elapsed = time.perf_counter() - t0

        n_frames = clip.get("n_frames", 0)
        total_frames += n_frames
        total_wall += elapsed

        tp, fp, fn = match_events(auto, human, tolerance)
        total_tp += tp
        total_fp += fp
        total_fn += fn

        if verbose:
            p = tp / (tp + fp) if (tp + fp) else 0
            r = tp / (tp + fn) if (tp + fn) else 0
            f1 = 2 * p * r / (p + r) if (p + r) else 0
            spd = n_frames / elapsed if elapsed else 0
            print(f"  {jp.stem}")
            print(f"    human={len(human)} auto={len(auto)}  TP={tp} FP={fp} FN={fn}"
                  f"  P={p:.2f} R={r:.2f} F1={f1:.2f}  {spd:.0f}fps")
            if auto:
                print(f"    auto:  {[f'{t:.2f}' for t in auto]}")
            print(f"    human: {[f'{t:.2f}' for t in human]}")
            print()

    if not (total_tp + total_fp + total_fn):
        print("No matching data — check annotations and config.", file=sys.stderr)
        return {}

    p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0
    r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0
    f1 = 2 * p * r / (p + r) if (p + r) else 0
    speed = total_frames / total_wall if total_wall else 0

    print("─" * 52)
    print(f"Clips evaluated : {len(clips)}")
    print(f"Human marks     : {total_tp + total_fn}")
    print(f"Auto events     : {total_tp + total_fp}")
    print(f"TP={total_tp}  FP={total_fp}  FN={total_fn}")
    print(f"Precision       : {p:.3f}")
    print(f"Recall          : {r:.3f}")
    print(f"F1              : {f1:.3f}")
    print(f"Speed           : {speed:.0f} fps")

    return {"p": p, "r": r, "f1": f1, "tp": total_tp, "fp": total_fp, "fn": total_fn}


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark detector against annotated clips")
    ap.add_argument("clips_dir", help="Directory containing clip MP4s and JSON sidecars")
    ap.add_argument("--config", default=None, help="Path to JSON config (for goal geometry + params)")
    ap.add_argument("--goal", default=None, help="Filter to a single goal name")
    ap.add_argument("--tolerance", type=float, default=1.0,
                    help="Match window in seconds (default: 1.0)")
    ap.add_argument("-v", "--verbose", action="store_true", help="Per-clip breakdown")
    # Allow overriding individual params for quick testing
    ap.add_argument("--ball-area", type=int, default=None)
    ap.add_argument("--band-width", type=int, default=None)
    ap.add_argument("--fall-ratio", type=float, default=None)
    ap.add_argument("--min-peak", type=int, default=None)
    ap.add_argument("--cooldown", type=int, default=None)
    args = ap.parse_args()

    overrides = {k: v for k, v in {
        "ball_area": args.ball_area,
        "band_width": args.band_width,
        "fall_ratio": args.fall_ratio,
        "min_peak": args.min_peak,
        "cooldown": args.cooldown,
    }.items() if v is not None}

    benchmark(
        Path(args.clips_dir),
        Path(args.config) if args.config else None,
        args.goal,
        args.tolerance,
        args.verbose,
        overrides,
    )


if __name__ == "__main__":
    main()
