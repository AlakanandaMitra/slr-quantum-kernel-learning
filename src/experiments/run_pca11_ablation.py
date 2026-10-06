#!/usr/bin/env python3
"""
PCA11 versus SLR ablation at 150 samples/group, five matched seeds.

Place this file in the same folder as:
  qka_gamma_only_soft_3group.py
  run_matched_multiseed_baselines.py

The script runs the PCA11 side only. Existing SLR results are reused.

PCA11 is defined as the first 11 coordinates of the already fitted PCA64
representation. This gives a same-dimensional unsupervised control and
isolates the effect of replacing PCA-only compression with PCA64->LDA11.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import SVC

import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.paths import RESULTS_DIR
from src.quantum import qka_gamma_only_soft_3group as qka
from src.classical import run_matched_multiseed_baselines as base


TRAIN_PER_GROUP = 150
N_DIM = 11

MATCHED_DIR = RESULTS_DIR / "matched_representation"
SLR_QKA_ROOT = RESULTS_DIR / "qka_gamma_only"
RESULTS_ROOT = RESULTS_DIR / "representation_ablation_pca11_vs_slr"

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument(
        "--models",
        nargs="+",
        choices=("rbf", "fixed_qk", "gamma_qka"),
        default=("rbf", "fixed_qk", "gamma_qka"),
    )
    p.add_argument("--matched-dir", type=Path, default=MATCHED_DIR)
    p.add_argument("--slr-qka-root", type=Path, default=SLR_QKA_ROOT)
    p.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    p.add_argument("--aer-device", choices=("GPU", "CPU"), default="GPU")
    p.add_argument("--precision", choices=("single", "double"), default="single")
    p.add_argument("--skip-existing", action="store_true")
    return p.parse_args()


def load_pca11_matched_split(matched_dir: Path, slr_qka_root: Path, seed: int):
    """
    Reuse EXACT train/val/test indices and labels from the existing SLR QKA run.
    """
    slr_name = f"qka_gamma_only_soft_3group_{TRAIN_PER_GROUP}_seed{seed}"
    slr_dir = slr_qka_root / slr_name

    needed = [
        matched_dir / "train_embeddings_pca.npy",
        matched_dir / "valid_embeddings_pca.npy",
        matched_dir / "test_embeddings_pca.npy",
        slr_dir / f"{slr_name}_train_indices_full.npy",
        slr_dir / f"{slr_name}_val_indices_full.npy",
        slr_dir / f"{slr_name}_test_indices_full.npy",
        slr_dir / f"{slr_name}_y_train_3group.npy",
        slr_dir / f"{slr_name}_y_val_3group.npy",
        slr_dir / f"{slr_name}_y_test_3group.npy",
    ]
    missing = [str(p) for p in needed if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing files:\n" + "\n".join(missing))

    Xtr64 = np.load(matched_dir / "train_embeddings_pca.npy")
    Xva64 = np.load(matched_dir / "valid_embeddings_pca.npy")
    Xte64 = np.load(matched_dir / "test_embeddings_pca.npy")

    itr = np.load(slr_dir / f"{slr_name}_train_indices_full.npy")
    iva = np.load(slr_dir / f"{slr_name}_val_indices_full.npy")
    ite = np.load(slr_dir / f"{slr_name}_test_indices_full.npy")

    arrays = {
        "X_train": Xtr64[itr, :N_DIM].astype(np.float32),
        "X_val": Xva64[iva, :N_DIM].astype(np.float32),
        "X_test": Xte64[ite, :N_DIM].astype(np.float32),
        "y_train": np.load(slr_dir / f"{slr_name}_y_train_3group.npy"),
        "y_val": np.load(slr_dir / f"{slr_name}_y_val_3group.npy"),
        "y_test": np.load(slr_dir / f"{slr_name}_y_test_3group.npy"),
    }

    print("PCA11 matched split")
    print(" train:", arrays["X_train"].shape,
          np.bincount(arrays["y_train"], minlength=3))
    print(" val  :", arrays["X_val"].shape,
          np.bincount(arrays["y_val"], minlength=3))
    print(" test :", arrays["X_test"].shape,
          np.bincount(arrays["y_test"], minlength=3))
    return arrays, itr, iva, ite


def run_gamma_qka(arrays, out_dir: Path, seed: int, device: str, precision: str):
    """
    Same gamma-only QKA protocol as qka_gamma_only_soft_3group.py,
    but using PCA11 instead of SLR.
    """
    run_name = f"pca11_gamma_only_qka_3group_{TRAIN_PER_GROUP}_seed{seed}"

    Xtr, Xva, Xte = arrays["X_train"], arrays["X_val"], arrays["X_test"]
    ytr, yva, yte = arrays["y_train"], arrays["y_val"], arrays["y_test"]

    # Same train-only [0,1] scaling as existing QKA.
    train_min = Xtr.min(axis=0)
    train_max = Xtr.max(axis=0)
    denom = np.maximum(train_max - train_min, 1e-12)
    scale = lambda X: np.clip((X - train_min) / denom, 0.0, 1.0)

    Xtr = scale(Xtr)
    Xva = scale(Xva)
    Xte = scale(Xte)

    np.save(out_dir / f"{run_name}_rescale_train_min.npy", train_min)
    np.save(out_dir / f"{run_name}_rescale_train_max.npy", train_max)

    feature_map, feature_params = qka.build_feature_map()
    backend = qka.get_aer_backend(device, precision)
    rng = np.random.default_rng(seed)

    objective = qka.GammaOnlyQKAObjective(
        Xtr, ytr, feature_map, feature_params, backend
    )

    total_start = time.perf_counter()

    best_gamma, best_C, best_iteration, spsa_time = qka.run_spsa(
        objective,
        rng,
        Xtr,
        ytr,
        Xva,
        yva,
        feature_map,
        feature_params,
        backend,
        out_dir,
        run_name,
    )

    kernel_start = time.perf_counter()
    sv_train = qka.compute_statevectors(
        Xtr, best_gamma, feature_map, feature_params, backend
    )
    sv_test = qka.compute_statevectors(
        Xte, best_gamma, feature_map, feature_params, backend
    )

    K_train_raw = qka.quantum_kernel_from_states(sv_train, sv_train)
    K_train = K_train_raw + qka.KERNEL_RIDGE * np.eye(
        len(ytr), dtype=np.float32
    )
    K_test_train = qka.quantum_kernel_from_states(sv_test, sv_train)
    kernel_time = time.perf_counter() - kernel_start

    clf = SVC(kernel="precomputed", C=best_C, class_weight="balanced")
    clf.fit(K_train, ytr)
    pred = clf.predict(K_test_train)

    final_alignment = qka.kernel_alignment(
        K_train_raw, qka.construct_soft_3group_target(ytr)
    )
    total_wall = time.perf_counter() - total_start

    np.save(out_dir / f"{run_name}_kernel_train_raw_final.npy", K_train_raw)
    np.save(out_dir / f"{run_name}_kernel_test_train_final.npy", K_test_train)
    np.save(out_dir / f"{run_name}_predictions.npy", pred)
    np.save(out_dir / f"{run_name}_best_gamma.npy", best_gamma)

    metrics = {
        "representation": "PCA11",
        "model": "Gamma-only QKA",
        "train_per_group": TRAIN_PER_GROUP,
        "seed": seed,
        "num_train": int(len(ytr)),
        "num_val": int(len(yva)),
        "num_test": int(len(yte)),
        "selected_C": float(best_C),
        "best_iteration": int(best_iteration),
        "test_accuracy": float(accuracy_score(yte, pred)),
        "test_macro_f1": float(f1_score(yte, pred, average="macro",
                                        zero_division=0)),
        "test_weighted_f1": float(
            f1_score(yte, pred, average="weighted", zero_division=0)
        ),
        "final_training_alignment": float(final_alignment),
        "trainable_parameter_count": 11,
        "objective_evaluations": int(objective.num_evaluations),
        "spsa_time_sec": float(spsa_time),
        "final_kernel_time_sec": float(kernel_time),
        "total_wall_time_sec": float(total_wall),
    }

    pd.DataFrame([metrics]).to_csv(
        out_dir / f"{run_name}_results_summary.csv", index=False
    )
    with open(out_dir / f"{run_name}_results_summary.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def main():
    args = parse_args()

    arrays, itr, iva, ite = load_pca11_matched_split(
        args.matched_dir, args.slr_qka_root, args.seed
    )

    out_dir = args.results_root / f"train150_seed{args.seed}" / "pca11"
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "train_indices_full.npy", itr)
    np.save(out_dir / "val_indices_full.npy", iva)
    np.save(out_dir / "test_indices_full.npy", ite)
    np.save(out_dir / "X_train_pca11.npy", arrays["X_train"])
    np.save(out_dir / "X_val_pca11.npy", arrays["X_val"])
    np.save(out_dir / "X_test_pca11.npy", arrays["X_test"])

    rows = []

    if "rbf" in args.models:
        target = out_dir / (
            f"rbf_svm_3group_{TRAIN_PER_GROUP}_seed{args.seed}"
            "_results_summary.json"
        )
        if not (args.skip_existing and target.exists()):
            m = base.run_rbf(arrays, out_dir, TRAIN_PER_GROUP, args.seed)
            m["representation"] = "PCA11"
            rows.append(m)

    if "fixed_qk" in args.models:
        target = out_dir / (
            f"fixed_qk_3group_{TRAIN_PER_GROUP}_seed{args.seed}"
            "_results_summary.json"
        )
        if not (args.skip_existing and target.exists()):
            m = base.run_fixed_qk(
                arrays, out_dir, TRAIN_PER_GROUP, args.seed,
                args.aer_device, args.precision
            )
            m["representation"] = "PCA11"
            rows.append(m)

    if "gamma_qka" in args.models:
        target = out_dir / (
            f"pca11_gamma_only_qka_3group_{TRAIN_PER_GROUP}_seed{args.seed}"
            "_results_summary.json"
        )
        if not (args.skip_existing and target.exists()):
            rows.append(
                run_gamma_qka(
                    arrays, out_dir, args.seed,
                    args.aer_device, args.precision
                )
            )

    if rows:
        pd.DataFrame(rows).to_csv(
            out_dir / "pca11_ablation_seed_summary.csv", index=False
        )
        print(pd.DataFrame(rows).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
