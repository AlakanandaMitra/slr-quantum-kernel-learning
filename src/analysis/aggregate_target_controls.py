#!/usr/bin/env python3
"""Aggregate soft, hard, and randomized gamma-QKA target-control results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd
from scipy.stats import wilcoxon


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--soft-root", type=Path, required=True)
    p.add_argument("--control-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--size", type=int, default=150)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 52, 62, 72, 82])
    return p.parse_args()


def main():
    args = parse_args()
    rows=[]
    for seed in args.seeds:
        for mode in ["soft", "hard", "randomized"]:
            run=f"qka_gamma_only_{mode}_3group_{args.size}_seed{seed}"
            root=args.soft_root if mode=="soft" else args.control_root
            path=root/run/f"{run}_results_summary.json"
            if not path.exists():
                print(f"WARNING missing: {path}")
                continue
            with open(path) as f: s=json.load(f)
            rows.append({
                "train_per_group": args.size,
                "seed": seed,
                "target_mode": mode,
                "test_accuracy": s["test_accuracy"],
                "test_macro_f1": s["test_macro_f1"],
                "test_weighted_f1": s["test_weighted_f1"],
                "final_training_alignment": s["final_training_alignment"],
                "total_wall_time_sec": s["total_wall_time_sec"],
            })
    df=pd.DataFrame(rows)
    if df.empty: raise RuntimeError("No target-control results found")
    summary=(df.groupby("target_mode",as_index=False)
             .agg(n_seeds=("seed","nunique"),
                  macro_f1_mean=("test_macro_f1","mean"),
                  macro_f1_sd=("test_macro_f1","std"),
                  accuracy_mean=("test_accuracy","mean"),
                  accuracy_sd=("test_accuracy","std"),
                  alignment_mean=("final_training_alignment","mean"),
                  alignment_sd=("final_training_alignment","std"),
                  runtime_mean_sec=("total_wall_time_sec","mean"),
                  runtime_sd_sec=("total_wall_time_sec","std")))
    tests=[]
    for a,b in [("soft","hard"),("soft","randomized"),("hard","randomized")]:
        pa=df[df.target_mode==a].set_index("seed")["test_macro_f1"]
        pb=df[df.target_mode==b].set_index("seed")["test_macro_f1"]
        common=pa.index.intersection(pb.index)
        stat,p=wilcoxon(pa.loc[common],pb.loc[common],alternative="two-sided")
        tests.append({"target_a":a,"target_b":b,"n":len(common),
                      "mean_difference_a_minus_b":(pa.loc[common]-pb.loc[common]).mean(),
                      "wilcoxon_statistic":stat,"wilcoxon_p_value":p})
    tests=pd.DataFrame(tests)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    df.to_csv(args.output_dir/"target_control_seed_results.csv",index=False)
    summary.to_csv(args.output_dir/"target_control_mean_std.csv",index=False)
    tests.to_csv(args.output_dir/"target_control_paired_tests.csv",index=False)
    print(summary.to_string(index=False))
    print("\nPaired tests:\n",tests.to_string(index=False))

if __name__=="__main__": main()
