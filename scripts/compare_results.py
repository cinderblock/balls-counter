#!/usr/bin/env python3
"""Compare benchmark results across detection methods.

Reads per-clip results saved by benchmark_ml.py and prints a comparison table.

Usage:
    uv run python scripts/compare_results.py
    uv run python scripts/compare_results.py data/benchmark_results.json
"""

import argparse
import json
import sys
from pathlib import Path


def print_table(results_data: dict) -> None:
    methods = results_data.get("methods", {})
    if not methods:
        print("No results found.")
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
    print(f"Ground truth: {human} marks across {n_clips} clips")
    updated = results_data.get("updated", "")
    if updated:
        print(f"Last updated: {updated}")
    print()

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


def main():
    ap = argparse.ArgumentParser(description="Compare benchmark results")
    ap.add_argument("results", nargs="?", default="data/benchmark_results.json",
                    help="Path to results JSON")
    args = ap.parse_args()

    path = Path(args.results)
    if not path.exists():
        print(f"No results file at {path}. Run benchmark_ml.py first.", file=sys.stderr)
        sys.exit(1)

    data = json.loads(path.read_text())

    # Handle old format (list of dicts) gracefully
    if isinstance(data, list):
        print("Results file is in old format. Re-run benchmark_ml.py to regenerate.", file=sys.stderr)
        sys.exit(1)

    print_table(data)


if __name__ == "__main__":
    main()
