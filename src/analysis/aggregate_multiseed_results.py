#!/usr/bin/env python3
"""Aggregate five-seed RBF-SVM, fixed-QK, and gamma-only-QKA results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys



REPO_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(REPO_ROOT))



from src.paths import RESULTS_DIR
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

DEFAULT_GAMMA_ROOT = RESULTS_DIR / "qka_gamma_only"

DEFAULT_BASELINE_ROOT = RESULTS_DIR / "multiseed_matched"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gamma-results-root", type=Path, default=DEFAULT_GAMMA_ROOT)
    p.add_argument("--baseline-results-root", type=Path, default=DEFAULT_BASELINE_ROOT)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_BASELINE_ROOT / "aggregate")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 52, 62, 72, 82])
    p.add_argument("--sizes", type=int, nargs="+", default=[100, 150, 210])
    return p.parse_args()


def read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    with open(path) as f:
        return json.load(f)


def collect(args) -> pd.DataFrame:
    rows = []
    for size in args.sizes:
        for seed in args.seeds:
            gamma_name = f"qka_gamma_only_soft_3group_{size}_seed{seed}"
            gamma_path = args.gamma_results_root / gamma_name / f"{gamma_name}_results_summary.json"
            if gamma_path.exists():
                d = read_json(gamma_path)
                d.update({"model": "Gamma-only QKA", "train_per_group": size, "seed": seed})
                rows.append(d)

            bdir = args.baseline_results_root / f"train{size}_seed{seed}"
            for prefix, model in (("rbf_svm", "RBF-SVM"), ("fixed_qk", "Fixed QK")):
                name = f"{prefix}_3group_{size}_seed{seed}"
                path = bdir / f"{name}_results_summary.json"
                if path.exists():
                    d = read_json(path)
                    d.update({"model": model, "train_per_group": size, "seed": seed})
                    rows.append(d)
    if not rows:
        raise RuntimeError("No completed result summaries were found.")
    return pd.DataFrame(rows)


def paired_tests(df: pd.DataFrame) -> pd.DataFrame:
    tests = []
    pairs = [("Gamma-only QKA", "Fixed QK"), ("Gamma-only QKA", "RBF-SVM"), ("Fixed QK", "RBF-SVM")]
    for size, sdf in df.groupby("train_per_group"):
        pivot = sdf.pivot_table(index="seed", columns="model", values="test_macro_f1", aggfunc="first")
        for a, b in pairs:
            if a not in pivot or b not in pivot:
                continue
            matched = pivot[[a, b]].dropna()
            diffs = matched[a] - matched[b]
            if len(matched) >= 2 and np.any(np.abs(diffs.to_numpy()) > 0):
                stat, p = wilcoxon(matched[a], matched[b], alternative="two-sided", zero_method="wilcox")
            else:
                stat, p = np.nan, np.nan
            tests.append({
                "train_per_group": size,
                "model_a": a,
                "model_b": b,
                "n_paired_seeds": len(matched),
                "mean_difference_a_minus_b": diffs.mean() if len(diffs) else np.nan,
                "wilcoxon_statistic": stat,
                "wilcoxon_p_value": p,
            })
    return pd.DataFrame(tests)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = collect(args)
    df.to_csv(args.output_dir / "all_seed_level_results.csv", index=False)

    metric_cols = ["test_accuracy", "test_macro_f1", "test_weighted_f1", "total_wall_time_sec"]
    existing = [c for c in metric_cols if c in df.columns]
    summary = (
        df.groupby(["train_per_group", "model"])[existing]
        .agg(["count", "mean", "std", "min", "max"])
        .reset_index()
    )
    summary.columns = ["_".join([str(x) for x in col if str(x)]) for col in summary.columns]
    summary.to_csv(args.output_dir / "mean_std_summary.csv", index=False)

    tests = paired_tests(df)
    tests.to_csv(args.output_dir / "paired_wilcoxon_tests.csv", index=False)

    print("\nSeed-level results:\n", df[["train_per_group", "seed", "model", "test_macro_f1"]].sort_values(["train_per_group", "seed", "model"]).to_string(index=False))
    print("\nMean/std summary:\n", summary.to_string(index=False))
    print("\nPaired tests:\n", tests.to_string(index=False))


if __name__ == "__main__":
    main()
