#!/usr/bin/env python3
"""Grid-search MotionCounter params over annotated clips, ranked by F1.

Usage:
    uv run python scripts/sweep_params.py configs/clips --config configs/live.json
    uv run python scripts/sweep_params.py configs/clips --config configs/live.json --goal red-goal --top 30
    uv run python scripts/sweep_params.py configs/clips --config configs/live.json --fine --workers 12
"""

import argparse
import itertools
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from ball_counter.config import load_configs
from benchmark import get_all_marks, load_annotated_clip, match_events, run_detector


PARAM_GRID: dict[str, list] = {
    "ball_area":  [400, 600, 900, 1200, 1600],
    "band_width": [10, 20, 30],
    "fall_ratio": [0.40, 0.50, 0.60, 0.70],
    "min_peak":   [0, 200, 500],
    "cooldown":   [0, 3, 5],
}

# Focused grid around best params from initial sweep
PARAM_GRID_FINE: dict[str, list] = {
    "ball_area":  [900],                            # doesn't affect results
    "band_width": [5, 8, 10, 12, 15],
    "fall_ratio": [0.35, 0.40, 0.45, 0.50, 0.55, 0.60],
    "min_peak":   [0, 100, 200, 300, 400, 500],
    "cooldown":   [4, 5, 6, 7, 8, 10],
}


def _eval_combo(combo_args: tuple) -> tuple:
    """Evaluate a single parameter combo against all clips. Runs in worker process."""
    params, clip_infos, goal_configs_ser, tolerance = combo_args

    # Reconstruct goal configs from serialized form
    from ball_counter.config import GoalConfig
    goal_configs = {name: GoalConfig(**d) for name, d in goal_configs_ser.items()}

    tp = fp = fn = 0
    for mp4_str, goal_name, human_times in clip_infos:
        gc = goal_configs.get(goal_name)
        if gc is None or not human_times:
            continue
        auto, _ = run_detector(Path(mp4_str), gc, 30.0, **params)
        _tp, _fp, _fn = match_events(auto, human_times, tolerance)
        tp += _tp
        fp += _fp
        fn += _fn

    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return (f1, p, r, tp, fp, fn, params)


def sweep(clips_dir: Path, config_path: Path, goal_filter: str | None,
          tolerance: float, top_n: int, grid: dict[str, list],
          workers: int = 1) -> None:
    goal_configs: dict = {}
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
        return

    keys = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    total = len(combos)
    print(f"Clips: {len(clips)}  Parameter combinations: {total}  Workers: {workers}\n")

    # Pre-compute human marks and serialize goal configs for worker processes
    clip_infos = []
    for jp, mp4, clip in clips:
        goal_name = clip.get("goal", "")
        human = get_all_marks(clip)
        clip_infos.append((str(mp4), goal_name, human))

    goal_configs_ser = {}
    for name, gc in goal_configs.items():
        goal_configs_ser[name] = {
            f.name: getattr(gc, f.name)
            for f in gc.__dataclass_fields__.values()
        }

    combo_args = [
        (dict(zip(keys, vals)), clip_infos, goal_configs_ser, tolerance)
        for vals in combos
    ]

    results = []

    if workers <= 1:
        # Sequential — same as before but with progress
        for i, args in enumerate(combo_args):
            results.append(_eval_combo(args))
            if (i + 1) % 25 == 0 or (i + 1) == total:
                best_f1 = max(r[0] for r in results)
                print(f"  {i + 1:>4}/{total}  best F1 so far: {best_f1:.3f}", end="\r")
    else:
        # Parallel
        done = 0
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_eval_combo, args): i for i, args in enumerate(combo_args)}
            for future in as_completed(futures):
                results.append(future.result())
                done += 1
                if done % 25 == 0 or done == total:
                    best_f1 = max(r[0] for r in results)
                    print(f"  {done:>4}/{total}  best F1 so far: {best_f1:.3f}", end="\r")

    print()
    results.sort(key=lambda r: (r[0], r[2], r[1]), reverse=True)

    col_w = 10
    header = (f"{'rank':>4}  {'F1':>6}  {'P':>6}  {'R':>6}  {'TP':>4}  {'FP':>4}  {'FN':>4}  "
              + "  ".join(f"{k:>{col_w}}" for k in keys))
    print(f"\nTop {min(top_n, len(results))} results:")
    print(header)
    print("─" * len(header))

    for rank, (f1, p, r, tp, fp, fn, params) in enumerate(results[:top_n], 1):
        row = (f"{rank:>4}  {f1:>6.3f}  {p:>6.3f}  {r:>6.3f}  {tp:>4}  {fp:>4}  {fn:>4}  "
               + "  ".join(f"{params[k]:>{col_w}}" for k in keys))
        print(row)

    if results:
        best = results[0]
        print(f"\nBest params (F1={best[0]:.3f}):")
        for k, v in best[6].items():
            print(f"  {k}: {v}")


def main() -> None:
    default_workers = max(1, (os.cpu_count() or 1) * 3 // 4)
    ap = argparse.ArgumentParser(description="Sweep MotionCounter params to maximise F1")
    ap.add_argument("clips_dir", help="Directory with annotated clips")
    ap.add_argument("--config", required=True, help="Path to JSON config")
    ap.add_argument("--goal", default=None, help="Filter to a single goal name")
    ap.add_argument("--tolerance", type=float, default=1.0, help="Match tolerance in seconds")
    ap.add_argument("--top", type=int, default=20, help="Show top N results")
    ap.add_argument("--fine", action="store_true", help="Use fine-grained grid around known best params")
    ap.add_argument("--workers", "-j", type=int, default=default_workers,
                    help=f"Number of parallel workers (default: {default_workers}, half of CPUs)")
    args = ap.parse_args()

    grid = PARAM_GRID_FINE if args.fine else PARAM_GRID
    sweep(
        Path(args.clips_dir),
        Path(args.config),
        args.goal,
        args.tolerance,
        args.top,
        grid,
        args.workers,
    )


if __name__ == "__main__":
    main()
