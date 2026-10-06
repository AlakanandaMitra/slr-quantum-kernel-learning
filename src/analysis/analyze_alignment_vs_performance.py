#!/usr/bin/env python3
"""Analyze gamma-QKA alignment versus validation/test performance across seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--qka-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 52, 62, 72, 82])
    p.add_argument("--sizes", type=int, nargs="+", default=[100, 150, 210])
    return p.parse_args()


def safe_corr(x: pd.Series, y: pd.Series, method: str) -> tuple[float, float]:
    mask = x.notna() & y.notna()
    xv, yv = x[mask].to_numpy(), y[mask].to_numpy()
    if len(xv) < 3 or np.allclose(xv, xv[0]) or np.allclose(yv, yv[0]):
        return np.nan, np.nan
    return pearsonr(xv, yv) if method == "pearson" else spearmanr(xv, yv)


def main() -> None:
    args = parse_args()
    rows = []

    for size in args.sizes:
        for seed in args.seeds:
            run = f"qka_gamma_only_soft_3group_{size}_seed{seed}"
            d = args.qka_root / run
            summary_path = d / f"{run}_results_summary.json"
            log_path = d / f"{run}_training_log_final.csv"
            if not summary_path.exists() or not log_path.exists():
                print(f"WARNING missing: {d}")
                continue

            with open(summary_path) as f:
                s = json.load(f)
            log = pd.read_csv(log_path)
            val_rows = log.dropna(subset=["val_macro_f1"])
            best_val = float(val_rows["val_macro_f1"].max()) if not val_rows.empty else np.nan
            initial_alignment = float(log.iloc[0]["alignment_current"])
            final_alignment = float(s["final_training_alignment"])

            rows.append({
                "train_per_group": size,
                "seed": seed,
                "initial_alignment": initial_alignment,
                "final_alignment": final_alignment,
                "alignment_gain": final_alignment - initial_alignment,
                "best_val_macro_f1": best_val,
                "test_macro_f1": float(s["test_macro_f1"]),
                "test_accuracy": float(s["test_accuracy"]),
                "best_iteration": int(s["best_iteration"]),
                "total_wall_time_sec": float(s["total_wall_time_sec"]),
            })

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No QKA runs found.")

    corr_rows = []
    for size_label, part in [("all", df)] + [(str(s), df[df.train_per_group == s]) for s in args.sizes]:
        for xcol in ["final_alignment", "alignment_gain"]:
            for ycol in ["best_val_macro_f1", "test_macro_f1"]:
                for method in ["pearson", "spearman"]:
                    r, p = safe_corr(part[xcol], part[ycol], method)
                    corr_rows.append({
                        "train_per_group": size_label,
                        "x": xcol,
                        "y": ycol,
                        "method": method,
                        "n": len(part),
                        "correlation": r,
                        "p_value": p,
                    })

    corrs = pd.DataFrame(corr_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_dir / "alignment_performance_seed_results.csv", index=False)
    corrs.to_csv(args.output_dir / "alignment_performance_correlations.csv", index=False)

    print(df.to_string(index=False))
    print("\nCorrelations:")
    print(corrs.to_string(index=False))
    print(f"\nSaved outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
