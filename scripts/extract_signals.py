#!/usr/bin/env python3
"""Extract per-frame signal traces from annotated clips for ML training.

Runs the existing MotionCounter signal pipeline (bg subtraction + color mask +
zone area) on each annotated clip and saves:
  - Per-frame signal values (the area time series)
  - Human mark timestamps converted to frame indices
  - Clip metadata (goal, fps, n_frames)

Output: a single .npz file with all traces, ready for train_peak_model.py.

Usage:
    uv run python scripts/extract_signals.py configs/clips --config configs/live.json -o data/signals.npz
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ball_counter.config import load_configs
from ball_counter.counter import MotionCounter


N_FEATURES = 6  # area, n_contours, centroid_x, centroid_y, bbox_w, bbox_h


def extract_signal(mp4_path: Path, goal_config, fps_hint: float = 30.0,
                   multi_channel: bool = False) -> tuple[np.ndarray, float]:
    """Run signal extraction on a clip.

    Returns (signal_array, fps).
    If multi_channel=False: signal_array is 1D (N_frames,) — area only.
    If multi_channel=True:  signal_array is 2D (N_frames, 6) — all features.
    """
    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        empty = np.array([], dtype=np.float32)
        return empty, fps_hint

    actual_fps = cap.get(cv2.CAP_PROP_FPS) or fps_hint
    ds = goal_config.downsample

    ret, first = cap.read()
    if not ret:
        cap.release()
        return np.array([], dtype=np.float32), actual_fps

    h, w = first.shape[:2]

    # Build counter (same as benchmark.py)
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
        line=line,
        roi=roi,
        ball_area=goal_config.ball_area,
        band_width=goal_config.band_width,
        min_peak=goal_config.min_peak,
        fall_ratio=goal_config.fall_ratio,
        cooldown=0,  # No cooldown — we want the raw signal
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
        arr = arr[:, 0]  # flatten back to 1D for backwards compat
    return arr, actual_fps


def get_all_marks(clip: dict, dedup_window: float = 0.15) -> list[float]:
    """Aggregate human marks, deduplicating accidental double-taps."""
    times = []
    for ann in clip["annotations"].values():
        for m in ann.get("marks", []):
            times.append(float(m["video_time"]))
    times.sort()
    deduped: list[float] = []
    for t in times:
        if not deduped or t - deduped[-1] > dedup_window:
            deduped.append(t)
    return deduped


def main():
    ap = argparse.ArgumentParser(description="Extract signal traces for ML training")
    ap.add_argument("clips_dir", help="Directory with clip MP4s + JSON sidecars")
    ap.add_argument("--config", required=True, help="Config JSON for goal geometry")
    ap.add_argument("-o", "--output", default="data/signals.npz", help="Output .npz path")
    ap.add_argument("--multi-channel", action="store_true",
                    help="Extract 6-channel features (area, n_contours, cx, cy, bw, bh)")
    args = ap.parse_args()

    clips_dir = Path(args.clips_dir)
    sources, _ = load_configs(Path(args.config))
    goal_configs = {}
    for src in sources:
        for g in src.goals:
            goal_configs[g.name] = g

    # Collect all clips
    all_signals = []      # list of 1D arrays
    all_labels = []        # list of 1D arrays (frame-level binary labels)
    all_clip_names = []
    all_fps = []
    all_mark_counts = []

    total_marks = 0
    total_frames = 0

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
        gc = goal_configs.get(goal_name)
        if gc is None:
            continue

        marks = get_all_marks(clip)
        if not marks:
            continue

        signal, fps = extract_signal(mp4, gc, clip.get("fps", 30.0),
                                     multi_channel=args.multi_channel)
        if len(signal) == 0:
            continue

        # Convert mark times to frame indices
        mark_frames = [int(round(t * fps)) for t in marks]
        mark_frames = [f for f in mark_frames if 0 <= f < len(signal)]

        # Build binary label array: 1 at mark frames, 0 elsewhere
        labels = np.zeros(len(signal), dtype=np.float32)
        for f in mark_frames:
            labels[f] = 1.0

        all_signals.append(signal)
        all_labels.append(labels)
        all_clip_names.append(jp.stem)
        all_fps.append(fps)
        all_mark_counts.append(len(mark_frames))

        total_marks += len(mark_frames)
        total_frames += len(signal)

        print(f"  {jp.stem:45s}  frames={len(signal):>5}  marks={len(mark_frames):>3}  fps={fps:.1f}")

    if not all_signals:
        print("No valid clips found.", file=sys.stderr)
        sys.exit(1)

    # Save as npz with variable-length arrays
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    n_channels = N_FEATURES if args.multi_channel else 1
    np.savez(
        out,
        # Variable-length arrays stored as object arrays
        signals=np.array(all_signals, dtype=object),
        labels=np.array(all_labels, dtype=object),
        clip_names=np.array(all_clip_names),
        fps=np.array(all_fps, dtype=np.float32),
        mark_counts=np.array(all_mark_counts, dtype=np.int32),
        n_channels=np.array(n_channels, dtype=np.int32),
    )

    ch_str = f"{n_channels}ch" if args.multi_channel else "1ch"
    print(f"\nSaved {len(all_signals)} clips, {total_frames} frames, {total_marks} marks, {ch_str} → {out}")


if __name__ == "__main__":
    main()
