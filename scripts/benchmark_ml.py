#!/usr/bin/env python3
"""Benchmark the ML peak detector against human annotations and the threshold baseline.

Runs both the threshold-based and ML-based peak detectors on annotated clips
and compares their performance side by side.

Usage:
    uv run python scripts/benchmark_ml.py configs/clips --config configs/live.json --model models/peak_detector.pt
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ball_counter.config import load_configs
from ball_counter.counter import MotionCounter
from ball_counter.ml_detector import MLPeakDetector


# ── Signal extraction (shared) ───────────────────────────────────────────────

def extract_signal_from_clip(mp4_path: Path, goal_config, fps_hint: float = 30.0
                             ) -> tuple[np.ndarray, float]:
    """Extract raw signal trace from a clip using the MotionCounter pipeline."""
    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        return np.array([], dtype=np.float32), fps_hint

    actual_fps = cap.get(cv2.CAP_PROP_FPS) or fps_hint
    ds = goal_config.downsample

    ret, first = cap.read()
    if not ret:
        cap.release()
        return np.array([], dtype=np.float32), actual_fps

    h, w = first.shape[:2]

    x1, y1 = 0, 0
    if goal_config.crop_override:
        x1, y1 = goal_config.crop_override[0], goal_config.crop_override[1]

    scaled_w = max(1, int(w * ds))
    scaled_h = max(1, int(h * ds))

    def offset_scale(pts):
        return [[int((p[0] - x1) * ds), int((p[1] - y1) * ds)] for p in pts]

    line = tuple(offset_scale(goal_config.line)) if goal_config.line else None
    roi = offset_scale(goal_config.roi_points) if goal_config.roi_points else None

    counter = MotionCounter(
        frame_shape=(scaled_h, scaled_w),
        line=line, roi=roi,
        ball_area=goal_config.ball_area,
        band_width=goal_config.band_width,
        min_peak=goal_config.min_peak,
        fall_ratio=goal_config.fall_ratio,
        cooldown=0,  # Raw signal, no cooldown
        hsv_low=goal_config.hsv_low,
        hsv_high=goal_config.hsv_high,
    )

    signals = []
    def feed(frame):
        if ds != 1.0:
            sw, sh = max(1, int(w * ds)), max(1, int(h * ds))
            frame = cv2.resize(frame, (sw, sh))
        counter.process_frame(frame)
        return counter.signal

    signals.append(feed(first))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        signals.append(feed(frame))

    cap.release()
    return np.array(signals, dtype=np.float32), actual_fps


# ── Threshold-based peak detection (existing method) ─────────────────────────

def detect_threshold(signal: np.ndarray, fps: float, goal_config,
                     **overrides) -> list[float]:
    """Apply the threshold peak detector to a signal trace."""
    ball_area = overrides.get("ball_area", goal_config.ball_area)
    fall_ratio = overrides.get("fall_ratio", goal_config.fall_ratio)
    cooldown = overrides.get("cooldown", goal_config.cooldown)
    min_rise = 100
    min_peak = overrides.get("min_peak", goal_config.min_peak)

    events = []
    rising = False
    peak_val = 0
    cd_remaining = 0
    prev_area = 0

    for i, area in enumerate(signal):
        area = float(area)
        if cd_remaining > 0:
            cd_remaining -= 1
        else:
            if area > prev_area and area > min_rise:
                rising = True
                peak_val = max(peak_val, area)
            elif rising and area < peak_val * fall_ratio:
                if peak_val >= min_peak:
                    n_balls = max(1, round(peak_val / ball_area))
                    t = i / fps
                    for _ in range(n_balls):
                        events.append(t)
                    cd_remaining = cooldown
                rising = False
                peak_val = 0
        prev_area = area

    return events


# ── ML-based peak detection (frame-by-frame, same as live) ───────────────────

def detect_ml(signal: np.ndarray, fps: float, ml_detector: MLPeakDetector
              ) -> list[float]:
    """Run the ML peak detector frame-by-frame, simulating the live path."""
    events: list[float] = []
    for i, val in enumerate(signal):
        ev = ml_detector.process_signal(int(val), i)
        if ev is not None:
            events.append(ev.frame / fps)
    # Drain any remaining pending events
    for i in range(100):
        ev = ml_detector.process_signal(0, len(signal) + i)
        if ev is not None:
            events.append(ev.frame / fps)
    return events


# ── Matching (same as benchmark.py) ──────────────────────────────────────────

def get_all_marks(clip: dict, dedup_window: float = 0.15) -> list[float]:
    times = []
    for ann in clip["annotations"].values():
        for m in ann.get("marks", []):
            times.append(float(m["video_time"]))
    times.sort()
    deduped = []
    for t in times:
        if not deduped or t - deduped[-1] > dedup_window:
            deduped.append(t)
    return deduped


def match_events(auto: list[float], human: list[float],
                 tolerance: float = 1.0) -> tuple[int, int, int]:
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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Benchmark ML vs threshold peak detection")
    ap.add_argument("clips_dir", help="Directory with clip MP4s + JSON sidecars")
    ap.add_argument("--config", required=True, help="Config JSON for goal geometry")
    ap.add_argument("--model", required=True, help="Path to trained .pt model")
    ap.add_argument("--goal", default=None, help="Filter to a single goal name")
    ap.add_argument("--tolerance", type=float, default=1.0, help="Match tolerance (seconds)")
    ap.add_argument("--threshold", type=float, default=0.5, help="ML detection threshold")
    ap.add_argument("--min-distance", type=int, default=3, help="Min frames between events")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--results", default=None,
                    help="Path to results JSON (default: data/benchmark_results.json)")
    # Threshold detector overrides
    ap.add_argument("--fall-ratio", type=float, default=None)
    ap.add_argument("--cooldown", type=int, default=None)
    args = ap.parse_args()

    clips_dir = Path(args.clips_dir)
    sources, _ = load_configs(Path(args.config))
    goal_configs = {}
    for src in sources:
        for g in src.goals:
            goal_configs[g.name] = g

    # Validate model path
    model_path = args.model
    if not Path(model_path).exists():
        print(f"Model not found: {model_path}", file=sys.stderr)
        sys.exit(1)
    print(f"Model: {model_path}")

    threshold_overrides = {}
    if args.fall_ratio is not None:
        threshold_overrides["fall_ratio"] = args.fall_ratio
    if args.cooldown is not None:
        threshold_overrides["cooldown"] = args.cooldown

    # Collect clips
    clips = []
    for jp in sorted(clips_dir.glob("*.json")):
        if jp.stem == "reviewers":
            continue
        mp4 = jp.with_suffix(".mp4")
        if not mp4.exists():
            continue
        clip = json.loads(jp.read_text())
        if not clip.get("annotations"):
            continue
        goal_name = clip.get("goal", "")
        if args.goal and goal_name != args.goal:
            continue
        gc = goal_configs.get(goal_name)
        if gc is None:
            continue
        marks = get_all_marks(clip)
        if not marks:
            continue
        clips.append((jp, mp4, clip, gc, marks))

    if not clips:
        print("No annotated clips found.", file=sys.stderr)
        sys.exit(1)

    print(f"Evaluating {len(clips)} clips  (tolerance={args.tolerance}s, "
          f"ml_threshold={args.threshold}, min_distance={args.min_distance})\n")

    # Accumulators
    thresh_tp = thresh_fp = thresh_fn = 0
    ml_tp = ml_fp = ml_fn = 0

    for jp, mp4, clip, gc, human in clips:
        signal, fps = extract_signal_from_clip(mp4, gc, clip.get("fps", 30.0))
        if len(signal) == 0:
            continue

        # Threshold detector
        auto_thresh = detect_threshold(signal, fps, gc, **threshold_overrides)
        t_tp, t_fp, t_fn = match_events(auto_thresh, human, args.tolerance)
        thresh_tp += t_tp
        thresh_fp += t_fp
        thresh_fn += t_fn

        # ML detector (fresh instance per clip, same as live stream)
        ml_det = MLPeakDetector(
            model_path,
            threshold=args.threshold,
            min_distance=args.min_distance,
            ball_area=gc.ball_area,
        )
        auto_ml = detect_ml(signal, fps, ml_det)
        m_tp, m_fp, m_fn = match_events(auto_ml, human, args.tolerance)
        ml_tp += m_tp
        ml_fp += m_fp
        ml_fn += m_fn

        if args.verbose:
            def fmt(tp, fp, fn):
                p = tp / max(1, tp + fp)
                r = tp / max(1, tp + fn)
                f1 = 2 * p * r / max(1e-8, p + r)
                return f"TP={tp} FP={fp} FN={fn}  P={p:.2f} R={r:.2f} F1={f1:.2f}"

            print(f"  {jp.stem}")
            print(f"    human={len(human)}")
            print(f"    thresh: auto={len(auto_thresh):>3}  {fmt(t_tp, t_fp, t_fn)}")
            print(f"    ml:     auto={len(auto_ml):>3}  {fmt(m_tp, m_fp, m_fn)}")
            if auto_ml:
                print(f"    ml_times:  {[f'{t:.2f}' for t in auto_ml]}")
            print(f"    human:     {[f'{t:.2f}' for t in human]}")
            print()

    human_total = thresh_tp + thresh_fn  # same for both

    def make_result(name, tp, fp, fn):
        p = tp / max(1, tp + fp)
        r = tp / max(1, tp + fn)
        f1 = 2 * p * r / max(1e-8, p + r)
        return {
            "method": name,
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1, 4),
            "human_marks": tp + fn,
            "n_clips": len(clips),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }

    results = [
        make_result("threshold", thresh_tp, thresh_fp, thresh_fn),
        make_result("ml-1dcnn", ml_tp, ml_fp, ml_fn),
    ]

    # Print summary
    for r in results:
        print(f"\n{'=' * 52}")
        print(f"  {r['method']}")
        print(f"  TP={r['tp']}  FP={r['fp']}  FN={r['fn']}")
        print(f"  Precision={r['precision']:.3f}  Recall={r['recall']:.3f}  F1={r['f1']:.3f}")

    # Save results
    results_path = Path(args.results) if args.results else None
    if results_path is None:
        results_path = Path("data/benchmark_results.json")

    results_path.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if results_path.exists():
        existing = json.loads(results_path.read_text())

    # Add params to each result for reproducibility
    for r in results:
        if r["method"] == "threshold":
            r["params"] = {
                "fall_ratio": threshold_overrides.get("fall_ratio", "config"),
                "cooldown": threshold_overrides.get("cooldown", "config"),
            }
        else:
            r["params"] = {
                "model": str(model_path),
                "threshold": args.threshold,
                "min_distance": args.min_distance,
            }

    existing.extend(results)
    results_path.write_text(json.dumps(existing, indent=2) + "\n")
    print(f"\nResults saved to {results_path} ({len(existing)} total entries)")


if __name__ == "__main__":
    main()
