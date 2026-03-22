#!/usr/bin/env python3
"""Benchmark detection methods against human annotations.

Stores per-clip results so new annotated clips can be incrementally
processed without re-running everything. Each method's results are
keyed by clip name — only missing clips are evaluated.

Usage:
    # Run all methods on any new/changed clips:
    uv run python scripts/benchmark_ml.py configs/clips --config configs/live.json --model models/peak_detector.pt

    # Force re-run everything:
    uv run python scripts/benchmark_ml.py configs/clips --config configs/live.json --model models/peak_detector.pt --force

    # Show comparison table from saved results:
    uv run python scripts/compare_results.py
"""

import argparse
import hashlib
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
from ball_counter.roi_detector import ROIBlobDetector


# ── Signal extraction ─────────────────────────────────────────────────────────

def extract_signal_from_clip(mp4_path: Path, goal_config, fps_hint: float = 30.0,
                             multi_channel: bool = False,
                             use_roi: bool = False) -> tuple[np.ndarray, float]:
    """Extract raw signal trace from a clip using the MotionCounter pipeline.

    If use_roi=True and roi_points exist, uses ROI ring mode instead of line band.
    """
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

    # Pick geometry: roi if requested and available, else line
    if use_roi and goal_config.roi_points:
        line = None
        roi = offset_scale(goal_config.roi_points)
    else:
        line = tuple(offset_scale(goal_config.line)) if goal_config.line else None
        roi = offset_scale(goal_config.roi_points) if (goal_config.roi_points and not line) else None

    counter = MotionCounter(
        frame_shape=(scaled_h, scaled_w),
        line=line, roi=roi,
        ball_area=goal_config.ball_area,
        band_width=goal_config.band_width,
        min_peak=goal_config.min_peak,
        fall_ratio=goal_config.fall_ratio,
        cooldown=0,
        hsv_low=goal_config.hsv_low,
        hsv_high=goal_config.hsv_high,
    )

    signals = []
    def feed(frame):
        if ds != 1.0:
            sw, sh = max(1, int(w * ds)), max(1, int(h * ds))
            frame = cv2.resize(frame, (sw, sh))
        counter.process_frame(frame)
        if multi_channel:
            return counter.signal_features
        return (counter.signal,)

    signals.append(feed(first))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        signals.append(feed(frame))

    cap.release()
    arr = np.array(signals, dtype=np.float32)
    if not multi_channel:
        arr = arr[:, 0]  # back to 1D
    return arr, actual_fps


# ── Detection methods ─────────────────────────────────────────────────────────

def detect_threshold(signal: np.ndarray, fps: float, goal_config,
                     **overrides) -> list[float]:
    """Threshold peak detector on signal trace."""
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


def detect_ml(signal: np.ndarray, fps: float, ml_detector: MLPeakDetector
              ) -> list[float]:
    """ML peak detector, frame-by-frame (same as live path)."""
    multi = signal.ndim == 2
    n_frames = len(signal)
    events: list[float] = []
    for i in range(n_frames):
        val = tuple(int(v) for v in signal[i]) if multi else int(signal[i])
        ev = ml_detector.process_signal(val, i)
        if ev is not None:
            events.append(ev.frame / fps)
    # Drain pending events
    zero = (0,) * (signal.shape[1] if multi else 1)
    for i in range(100):
        ev = ml_detector.process_signal(zero if multi else 0, n_frames + i)
        if ev is not None:
            events.append(ev.frame / fps)
    return events


def detect_roi_blob(mp4_path: Path, goal_config, fps_hint: float = 30.0
                    ) -> tuple[list[float], float]:
    """Run ROI blob detector on a clip. Returns (event_times, fps)."""
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
    x1, y1 = 0, 0
    if goal_config.crop_override:
        x1, y1 = goal_config.crop_override[0], goal_config.crop_override[1]

    scaled_w = max(1, int(w * ds))
    scaled_h = max(1, int(h * ds))

    def offset_scale(pts):
        return [[int((p[0] - x1) * ds), int((p[1] - y1) * ds)] for p in pts]

    roi = offset_scale(goal_config.roi_points)

    detector = ROIBlobDetector(
        frame_shape=(scaled_h, scaled_w),
        roi_points=roi,
        hsv_low=goal_config.hsv_low,
        hsv_high=goal_config.hsv_high,
        min_area=50,
    )

    def feed(frame):
        if ds != 1.0:
            sw, sh = max(1, int(w * ds)), max(1, int(h * ds))
            frame = cv2.resize(frame, (sw, sh))
        return detector.process_frame(frame)

    feed(first)  # init bg model
    event_times = []
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


def detect_hybrid(signal: np.ndarray, fps: float, goal_config,
                  ml_detector: MLPeakDetector, tolerance: float = 0.5,
                  **threshold_overrides) -> list[float]:
    """Hybrid: threshold for immediate detections, ML fills in what it missed.

    1. Run threshold detector → high-confidence events
    2. Run ML detector frame-by-frame → all events including close ones
    3. Keep all threshold events
    4. Add ML events that aren't near any threshold event (the missed ones)
    """
    sig_1d = signal[:, 0] if signal.ndim == 2 else signal
    thresh_events = detect_threshold(sig_1d, fps, goal_config, **threshold_overrides)
    ml_events = detect_ml(signal, fps, ml_detector)

    # Start with all threshold events
    combined = list(thresh_events)

    # Add ML events that threshold missed
    for ml_t in ml_events:
        near_thresh = any(abs(ml_t - te) < tolerance for te in thresh_events)
        if not near_thresh:
            combined.append(ml_t)

    combined.sort()
    return combined


# ── Matching ──────────────────────────────────────────────────────────────────

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


def clip_fingerprint(clip: dict) -> str:
    """Hash of clip annotations so we can detect re-annotated clips."""
    raw = json.dumps(clip.get("annotations", {}), sort_keys=True)
    return hashlib.md5(raw.encode()).hexdigest()[:12]


# ── Results storage ───────────────────────────────────────────────────────────

def load_results(path: Path) -> dict:
    """Load results file. Structure:

    {
        "methods": {
            "threshold": {
                "params": {...},
                "clips": {
                    "clip_name": {"tp": N, "fp": N, "fn": N, "human": N, "fingerprint": "abc123"},
                    ...
                }
            },
            ...
        }
    }
    """
    if path.exists():
        return json.loads(path.read_text())
    return {"methods": {}}


def save_results(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def get_stale_clips(method_data: dict, all_clips: dict[str, str]) -> list[str]:
    """Return clip names that need (re-)evaluation.

    A clip needs evaluation if:
    - It's not in the saved results
    - Its annotation fingerprint has changed
    """
    saved = method_data.get("clips", {})
    stale = []
    for clip_name, fingerprint in all_clips.items():
        existing = saved.get(clip_name)
        if existing is None or existing.get("fingerprint") != fingerprint:
            stale.append(clip_name)
    return stale


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Benchmark detection methods (incremental)")
    ap.add_argument("clips_dir", help="Directory with clip MP4s + JSON sidecars")
    ap.add_argument("--config", required=True, help="Config JSON for goal geometry")
    ap.add_argument("--model", default=None, help="Path to trained .pt model (enables ML method)")
    ap.add_argument("--goal", default=None, help="Filter to a single goal name")
    ap.add_argument("--tolerance", type=float, default=1.0, help="Match tolerance (seconds)")
    ap.add_argument("--threshold", type=float, default=0.6, help="ML detection threshold")
    ap.add_argument("--min-distance", type=int, default=3, help="Min frames between ML events")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--results", default="data/benchmark_results.json",
                    help="Path to results JSON")
    ap.add_argument("--force", action="store_true", help="Re-run all clips, ignoring cache")
    ap.add_argument("--fall-ratio", type=float, default=None)
    ap.add_argument("--cooldown", type=int, default=None)
    args = ap.parse_args()

    clips_dir = Path(args.clips_dir)
    sources, _ = load_configs(Path(args.config))
    goal_configs = {}
    for src in sources:
        for g in src.goals:
            goal_configs[g.name] = g

    threshold_overrides = {}
    if args.fall_ratio is not None:
        threshold_overrides["fall_ratio"] = args.fall_ratio
    if args.cooldown is not None:
        threshold_overrides["cooldown"] = args.cooldown

    # Discover all annotated clips
    all_clips: dict[str, dict] = {}  # clip_name -> {jp, mp4, clip, gc, marks, fingerprint}
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
        all_clips[jp.stem] = {
            "jp": jp, "mp4": mp4, "clip": clip, "gc": gc,
            "marks": marks, "fingerprint": clip_fingerprint(clip),
        }

    if not all_clips:
        print("No annotated clips found.", file=sys.stderr)
        sys.exit(1)

    # Load existing results
    results_path = Path(args.results)
    results_data = load_results(results_path)
    if args.force:
        results_data = {"methods": {}}

    # Check if any goal has roi_points
    has_roi = any(gc.roi_points for gc in goal_configs.values())

    # Define methods to benchmark
    ml_channels = 1
    methods = {
        "threshold": {
            "params": {
                "fall_ratio": threshold_overrides.get("fall_ratio", "config"),
                "cooldown": threshold_overrides.get("cooldown", "config"),
            },
        },
    }
    if has_roi:
        methods["threshold-roi"] = {
            "params": {
                "geometry": "roi_points (ring mask + peak detect)",
                "fall_ratio": threshold_overrides.get("fall_ratio", "config"),
                "cooldown": threshold_overrides.get("cooldown", "config"),
            },
        }
        methods["roi-blob"] = {
            "params": {
                "geometry": "roi_points (blob tracking, outlet mode)",
                "min_area": 50,
            },
        }
    if args.model:
        model_path = Path(args.model)
        if not model_path.exists():
            print(f"Model not found: {model_path}", file=sys.stderr)
            sys.exit(1)
        # Peek at model to get channel count
        import torch as _torch
        _ckpt = _torch.load(str(model_path), map_location="cpu", weights_only=True)
        ml_channels = _ckpt.get("in_channels", 1)
        del _ckpt

        method_label = f"ml-{ml_channels}ch" if ml_channels > 1 else "ml-1dcnn"
        methods[method_label] = {
            "params": {
                "model": str(model_path),
                "threshold": args.threshold,
                "min_distance": args.min_distance,
                "channels": ml_channels,
            },
        }
        hybrid_label = f"hybrid-{ml_channels}ch" if ml_channels > 1 else "hybrid"
        methods[hybrid_label] = {
            "params": {
                "model": str(model_path),
                "threshold": args.threshold,
                "min_distance": args.min_distance,
                "channels": ml_channels,
                "strategy": "threshold + ML fill-in",
            },
        }

    # Fingerprints for all current clips
    clip_fingerprints = {name: info["fingerprint"] for name, info in all_clips.items()}

    # Check what needs updating across all methods
    any_work = False
    for method_name, method_info in methods.items():
        if method_name not in results_data["methods"]:
            results_data["methods"][method_name] = {"params": method_info["params"], "clips": {}}
        stale = get_stale_clips(results_data["methods"][method_name], clip_fingerprints)
        if stale:
            any_work = True
            break

    if not any_work and not args.force:
        print(f"No new clips. {len(all_clips)} annotated clips, all up to date.\n")
    else:
        total_new = set()
        for method_name in methods:
            stale = get_stale_clips(results_data["methods"].get(method_name, {}), clip_fingerprints)
            total_new.update(stale)
        new_marks = sum(len(all_clips[c]["marks"]) for c in total_new if c in all_clips)
        print(f"Updating with {len(total_new)} new/changed clips ({new_marks} marks). "
              f"{len(all_clips)} total annotated clips.\n")

    # Process each method
    for method_name, method_info in methods.items():
        if method_name not in results_data["methods"]:
            results_data["methods"][method_name] = {"params": method_info["params"], "clips": {}}
        method_data = results_data["methods"][method_name]
        method_data["params"] = method_info["params"]

        # Remove clips no longer in the dataset
        saved_clips = set(method_data.get("clips", {}).keys())
        current_clips = set(clip_fingerprints.keys())
        removed = saved_clips - current_clips
        for r in removed:
            del method_data["clips"][r]

        stale = get_stale_clips(method_data, clip_fingerprints)

        if not stale:
            continue

        print(f"  {method_name}: evaluating {len(stale)} clips "
              f"({len(current_clips) - len(stale)} cached)")

        needs_ml = method_name.startswith("ml") or method_name.startswith("hybrid")
        need_multi = needs_ml and args.model and ml_channels > 1
        needs_roi = method_name.endswith("-roi") or method_name == "roi-blob"
        needs_video = method_name == "roi-blob"  # needs raw frames, not signal

        for clip_name in stale:
            info = all_clips[clip_name]
            # Skip ROI methods for goals without roi_points
            if needs_roi and not info["gc"].roi_points:
                continue

            if needs_video:
                # ROI blob runs directly on video frames
                auto, fps = detect_roi_blob(
                    info["mp4"], info["gc"], info["clip"].get("fps", 30.0))
            else:
                signal, fps = extract_signal_from_clip(
                    info["mp4"], info["gc"], info["clip"].get("fps", 30.0),
                    multi_channel=need_multi, use_roi=needs_roi)
                if len(signal) == 0:
                    continue

                if method_name in ("threshold", "threshold-roi"):
                    sig_1d = signal[:, 0] if signal.ndim == 2 else signal
                    auto = detect_threshold(sig_1d, fps, info["gc"], **threshold_overrides)
                elif method_name.startswith("hybrid"):
                    ml_det = MLPeakDetector(
                        str(model_path),
                        threshold=args.threshold,
                        min_distance=args.min_distance,
                        ball_area=info["gc"].ball_area,
                    )
                    auto = detect_hybrid(signal, fps, info["gc"], ml_det,
                                         tolerance=0.3, **threshold_overrides)
                elif method_name.startswith("ml"):
                    ml_det = MLPeakDetector(
                        str(model_path),
                        threshold=args.threshold,
                        min_distance=args.min_distance,
                        ball_area=info["gc"].ball_area,
                    )
                    auto = detect_ml(signal, fps, ml_det)
                else:
                    continue

            tp, fp, fn = match_events(auto, info["marks"], args.tolerance)
            method_data["clips"][clip_name] = {
                "tp": tp, "fp": fp, "fn": fn,
                "human": len(info["marks"]),
                "fingerprint": info["fingerprint"],
            }

            if args.verbose:
                p = tp / max(1, tp + fp)
                r = tp / max(1, tp + fn)
                f1 = 2 * p * r / max(1e-8, p + r)
                print(f"  {clip_name}: human={len(info['marks'])} auto={tp+fp} "
                      f"TP={tp} FP={fp} FN={fn} F1={f1:.2f}")

    # Save
    results_data["updated"] = datetime.now().isoformat(timespec="seconds")
    save_results(results_path, results_data)

    # Print summary
    print(f"\nResults saved to {results_path}")
    print()

    # Inline comparison
    _print_comparison(results_data)


def _print_comparison(results_data: dict) -> None:
    """Print comparison table from results data."""
    methods = results_data.get("methods", {})
    if not methods:
        return

    rows = []
    for method_name, method_data in methods.items():
        clips = method_data.get("clips", {})
        tp = sum(c["tp"] for c in clips.values())
        fp = sum(c["fp"] for c in clips.values())
        fn = sum(c["fn"] for c in clips.values())
        human = sum(c["human"] for c in clips.values())
        p = tp / max(1, tp + fp)
        r = tp / max(1, tp + fn)
        f1 = 2 * p * r / max(1e-8, p + r)
        rows.append({
            "method": method_name, "tp": tp, "fp": fp, "fn": fn,
            "human": human, "n_clips": len(clips),
            "precision": p, "recall": r, "f1": f1,
            "params": method_data.get("params", {}),
        })

    if not rows:
        return

    human = rows[0]["human"]
    n_clips = rows[0]["n_clips"]
    print(f"Ground truth: {human} marks across {n_clips} clips\n")

    name_w = max(len(r["method"]) for r in rows)
    name_w = max(name_w, 6)

    header = (f"{'Method':<{name_w}}  "
              f"{'TP':>12}  {'FP':>10}  {'FN':>10}  "
              f"{'Prec':>14}  {'Recall':>14}  {'F1':>14}")
    print(header)
    print("-" * len(header))

    for r in rows:
        def fmt_int(val, delta):
            s = f"{val}"
            if delta != 0:
                s += f" ({delta:+d})"
            return s

        def fmt_float(val, delta):
            s = f"{val:.3f}"
            if abs(delta) > 0.0005:
                s += f" ({delta:+.3f})"
            return s

        print(f"{r['method']:<{name_w}}  "
              f"{fmt_int(r['tp'], r['tp'] - r['human']):>12}  "
              f"{fmt_int(r['fp'], r['fp']):>10}  "
              f"{fmt_int(r['fn'], r['fn']):>10}  "
              f"{fmt_float(r['precision'], r['precision'] - 1.0):>14}  "
              f"{fmt_float(r['recall'], r['recall'] - 1.0):>14}  "
              f"{fmt_float(r['f1'], r['f1'] - 1.0):>14}")

    print()
    for r in rows:
        ps = ", ".join(f"{k}={v}" for k, v in r["params"].items())
        print(f"  {r['method']}: {ps}")


if __name__ == "__main__":
    main()
