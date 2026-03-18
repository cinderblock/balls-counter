#!/usr/bin/env python3
"""Grid-search MotionCounter params over annotated clips, ranked by F1.

Usage:
    uv run python scripts/sweep_params.py configs/clips --config configs/live.json
    uv run python scripts/sweep_params.py configs/clips --config configs/live.json --goal red-goal --top 30
"""

import argparse
import copy
import itertools
import sys
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


def sweep(clips_dir: Path, config_path: Path, goal_filter: str | None,
          tolerance: float, top_n: int, grid: dict[str, list]) -> None:
    goal_configs: dict = {}
    for src in load_configs(config_path):
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
    print(f"Clips: {len(clips)}  Parameter combinations: {total}\n")

    results = []
    for i, vals in enumerate(combos):
        params = dict(zip(keys, vals))
        tp = fp = fn = 0

        for jp, mp4, clip in clips:
            gc = goal_configs.get(clip.get("goal", ""))
            if gc is None:
                continue
            human = get_all_marks(clip)
            if not human:
                continue
            auto, _ = run_detector(mp4, gc, clip.get("fps", 30.0), **params)
            _tp, _fp, _fn = match_events(auto, human, tolerance)
            tp += _tp
            fp += _fp
            fn += _fn

        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        results.append((f1, p, r, tp, fp, fn, params))

        if (i + 1) % 25 == 0 or (i + 1) == total:
            best_f1 = max(r[0] for r in results) if results else 0
            print(f"  {i + 1:>4}/{total}  best F1 so far: {best_f1:.3f}", end="\r")

    print()
    results.sort(reverse=True)

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
    ap = argparse.ArgumentParser(description="Sweep MotionCounter params to maximise F1")
    ap.add_argument("clips_dir", help="Directory with annotated clips")
    ap.add_argument("--config", required=True, help="Path to JSON config")
    ap.add_argument("--goal", default=None, help="Filter to a single goal name")
    ap.add_argument("--tolerance", type=float, default=1.0, help="Match tolerance in seconds")
    ap.add_argument("--top", type=int, default=20, help="Show top N results")
    args = ap.parse_args()

    sweep(
        Path(args.clips_dir),
        Path(args.config),
        args.goal,
        args.tolerance,
        args.top,
        PARAM_GRID,
    )


if __name__ == "__main__":
    main()
