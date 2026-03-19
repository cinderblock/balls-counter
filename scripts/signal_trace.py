#!/usr/bin/env python3
"""Dump per-frame signal trace for a clip, showing where events fire.

Usage:
    uv run python scripts/signal_trace.py configs/clips/20260318_114351_red-goal.mp4 --config configs/live.json --goal red-goal
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ball_counter.config import load_configs
from ball_counter.counter import MotionCounter, MotionEvent


def trace(mp4_path: Path, goal_config, fps_hint: float = 30.0,
          **param_overrides) -> None:
    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        print("Cannot open", mp4_path)
        return

    actual_fps = cap.get(cv2.CAP_PROP_FPS) or fps_hint
    ds = goal_config.downsample

    ret, first = cap.read()
    if not ret:
        cap.release()
        return

    h, w = first.shape[:2]

    # Build counter with same logic as benchmark
    x1, y1 = 0, 0
    if goal_config.crop_override:
        x1, y1 = goal_config.crop_override[0], goal_config.crop_override[1]

    scaled_w = max(1, int(w * ds))
    scaled_h = max(1, int(h * ds))

    def offset_scale(pts):
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

    counter = MotionCounter(frame_shape=(scaled_h, scaled_w), line=line, roi=roi, **params)

    def feed(frame):
        if ds != 1.0:
            sw, sh = max(1, int(w * ds)), max(1, int(h * ds))
            frame = cv2.resize(frame, (sw, sh))
        return counter.process_frame(frame)

    feed(first)
    frame_idx = 1

    print(f"{'frame':>6}  {'time':>7}  {'signal':>7}  {'rising':>6}  {'cd':>3}  event")
    print("-" * 60)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        ev = feed(frame)
        t = frame_idx / actual_fps
        sig = counter.signal
        rising = counter.rising
        cd = counter.cooldown_remaining

        # Only print frames with signal > 0 or events
        if sig > 0 or ev is not None:
            ev_str = f"  ** {ev.n_balls} ball(s) peak={ev.peak_area}" if ev else ""
            print(f"{frame_idx:>6}  {t:>7.2f}  {sig:>7}  {str(rising):>6}  {cd:>3}{ev_str}")

        frame_idx += 1

    cap.release()
    print(f"\nTotal events: {counter.count}")


def main():
    ap = argparse.ArgumentParser(description="Dump signal trace for a clip")
    ap.add_argument("mp4", help="Path to MP4 clip")
    ap.add_argument("--config", required=True, help="Config JSON for goal geometry")
    ap.add_argument("--goal", required=True, help="Goal name to use from config")
    ap.add_argument("--band-width", type=int, default=None)
    ap.add_argument("--fall-ratio", type=float, default=None)
    ap.add_argument("--min-peak", type=int, default=None)
    ap.add_argument("--cooldown", type=int, default=None)
    args = ap.parse_args()

    sources, _ = load_configs(Path(args.config))
    gc = None
    for src in sources:
        for g in src.goals:
            if g.name == args.goal:
                gc = g
                break

    if gc is None:
        print(f"Goal '{args.goal}' not found in config")
        sys.exit(1)

    overrides = {k: v for k, v in {
        "band_width": args.band_width,
        "fall_ratio": args.fall_ratio,
        "min_peak": args.min_peak,
        "cooldown": args.cooldown,
    }.items() if v is not None}

    trace(Path(args.mp4), gc, **overrides)


if __name__ == "__main__":
    main()
