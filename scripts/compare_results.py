#!/usr/bin/env python3
"""Compare benchmark results across detection methods.

Reads saved results from benchmark_ml.py and prints a comparison table.
Shows the latest run for each method by default, or all runs with --all.

Usage:
    uv run python scripts/compare_results.py
    uv run python scripts/compare_results.py --all
    uv run python scripts/compare_results.py data/benchmark_results.json
"""

import argparse
import json
import sys
from pathlib import Path


def print_table(results: list[dict]) -> None:
    if not results:
        print("No results found.")
        return

    human = results[0]["human_marks"]
    print(f"Ground truth: {human} marks across {results[0]['n_clips']} clips\n")

    # Column widths
    name_w = max(len(r["method"]) for r in results)
    name_w = max(name_w, 6)  # "Method"

    # Header
    header = (f"{'Method':<{name_w}}  "
              f"{'TP':>12}  {'FP':>10}  {'FN':>10}  "
              f"{'Prec':>14}  {'Recall':>14}  {'F1':>14}  "
              f"{'Date':>10}")
    print(header)
    print("-" * len(header))

    for r in results:
        gt = r["human_marks"]
        tp_d = r["tp"] - gt
        fp_d = r["fp"]
        fn_d = r["fn"]
        p_d = r["precision"] - 1.0
        r_d = r["recall"] - 1.0
        f1_d = r["f1"] - 1.0

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

        date = r.get("timestamp", "")[:10]

        print(f"{r['method']:<{name_w}}  "
              f"{fmt_int(r['tp'], tp_d):>12}  "
              f"{fmt_int(r['fp'], fp_d):>10}  "
              f"{fmt_int(r['fn'], fn_d):>10}  "
              f"{fmt_float(r['precision'], p_d):>14}  "
              f"{fmt_float(r['recall'], r_d):>14}  "
              f"{fmt_float(r['f1'], f1_d):>14}  "
              f"{date:>10}")

    # Params detail
    print()
    for r in results:
        params = r.get("params", {})
        if params:
            ps = ", ".join(f"{k}={v}" for k, v in params.items())
            print(f"  {r['method']}: {ps}")


def main():
    ap = argparse.ArgumentParser(description="Compare benchmark results")
    ap.add_argument("results", nargs="?", default="data/benchmark_results.json",
                    help="Path to results JSON")
    ap.add_argument("--all", action="store_true", help="Show all runs, not just latest per method")
    args = ap.parse_args()

    path = Path(args.results)
    if not path.exists():
        print(f"No results file at {path}. Run benchmark_ml.py first.", file=sys.stderr)
        sys.exit(1)

    all_results = json.loads(path.read_text())

    if args.all:
        print_table(all_results)
    else:
        # Latest per method
        latest: dict[str, dict] = {}
        for r in all_results:
            latest[r["method"]] = r
        print_table(list(latest.values()))


if __name__ == "__main__":
    main()
