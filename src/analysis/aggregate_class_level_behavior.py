#!/usr/bin/env python3
"""Aggregate Healthy/Single/Multiple metrics across seeds and training sizes."""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

CLASSES = ["Healthy", "Single", "Multiple"]
MODELS = {
    "Gamma-only QKA": "qka_gamma_only_soft_3group_{size}_seed{seed}",
    "RBF-SVM": "rbf_svm_3group_{size}_seed{seed}",
    "Fixed QK": "fixed_qk_3group_{size}_seed{seed}",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--qka-root", type=Path, required=True)
    p.add_argument("--baseline-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 52, 62, 72, 82])
    p.add_argument("--sizes", type=int, nargs="+", default=[100, 150, 210])
    return p.parse_args()


def report_path(model: str, size: int, seed: int, qka_root: Path, baseline_root: Path) -> Path:
    run = MODELS[model].format(size=size, seed=seed)
    if model == "Gamma-only QKA":
        return qka_root / run / f"{run}_classification_report.csv"
    return baseline_root / f"train{size}_seed{seed}" / f"{run}_classification_report.csv"


def main() -> None:
    args = parse_args()
    rows = []
    missing = []

    for size in args.sizes:
        for seed in args.seeds:
            for model in MODELS:
                path = report_path(model, size, seed, args.qka_root, args.baseline_root)
                if not path.exists():
                    missing.append(str(path))
                    continue
                df = pd.read_csv(path, index_col=0)
                for cls in CLASSES:
                    if cls not in df.index:
                        raise ValueError(f"{cls!r} missing from {path}")
                    rows.append({
                        "train_per_group": size,
                        "seed": seed,
                        "model": model,
                        "class": cls,
                        "precision": float(df.loc[cls, "precision"]),
                        "recall": float(df.loc[cls, "recall"]),
                        "f1_score": float(df.loc[cls, "f1-score"]),
                        "support": float(df.loc[cls, "support"]),
                    })

    if missing:
        print("WARNING: missing files:")
        for item in missing:
            print("  ", item)

    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("No classification reports were found.")

    summary = (
        out.groupby(["train_per_group", "model", "class"], as_index=False)
        .agg(
            n_seeds=("seed", "nunique"),
            precision_mean=("precision", "mean"),
            precision_sd=("precision", "std"),
            recall_mean=("recall", "mean"),
            recall_sd=("recall", "std"),
            f1_mean=("f1_score", "mean"),
            f1_sd=("f1_score", "std"),
            support_mean=("support", "mean"),
        )
        .sort_values(["train_per_group", "model", "class"])
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_dir / "class_level_seed_results.csv", index=False)
    summary.to_csv(args.output_dir / "class_level_mean_std.csv", index=False)

    pivot = summary.pivot_table(
        index=["train_per_group", "model"],
        columns="class",
        values="f1_mean",
    ).reset_index()
    pivot.to_csv(args.output_dir / "class_level_f1_compact.csv", index=False)

    print(summary.to_string(index=False))
    print(f"\nSaved outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
