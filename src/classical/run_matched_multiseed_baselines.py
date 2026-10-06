#!/usr/bin/env python3
"""Run matched RBF-SVM and fixed quantum-kernel baselines on the exact
train/validation/test partitions saved by qka_gamma_only_soft_3group.py.

The script is intentionally separate from the gamma-only QKA code.  Run the
QKA script first for a given training size and seed; this script then reuses
its saved split arrays so every model is evaluated on identical samples.
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.paths import RESULTS_DIR
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.svm import SVC

N_QUBITS = 11
FEATURE_MAP_REPS = 3
ENTANGLEMENT = "circular"
KERNEL_RIDGE = 0.1
SVC_C_GRID = [0.1, 1.0, 10.0, 100.0]
RBF_GAMMA_GRID: list[Any] = ["scale", 0.01, 0.1, 1.0]
GROUP_NAMES = {0: "Healthy", 1: "Single", 2: "Multiple"}

DEFAULT_GAMMA_ROOT = RESULTS_DIR / "qka_gamma_only"
DEFAULT_RESULTS_ROOT = RESULTS_DIR / "multiseed_matched"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-per-group", type=int, choices=(100, 150, 210), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--gamma-results-root", type=Path, default=DEFAULT_GAMMA_ROOT)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--models", nargs="+", choices=("rbf", "fixed_qk"), default=("rbf", "fixed_qk"))
    parser.add_argument("--aer-device", choices=("GPU", "CPU"), default="GPU")
    parser.add_argument("--precision", choices=("single", "double"), default="single")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def get_nvidia_smi_text() -> str:
    try:
        p = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=False)
        return p.stdout if p.returncode == 0 else p.stderr
    except Exception as exc:
        return repr(exc)


def gamma_run_paths(root: Path, train_per_group: int, seed: int) -> tuple[Path, str]:
    run_name = f"qka_gamma_only_soft_3group_{train_per_group}_seed{seed}"
    return root / run_name, run_name


def load_saved_split(root: Path, train_per_group: int, seed: int):
    run_dir, run_name = gamma_run_paths(root, train_per_group, seed)
    paths = {
        "X_train": run_dir / f"{run_name}_X_train_pca64_lda11.npy",
        "X_val": run_dir / f"{run_name}_X_val_pca64_lda11.npy",
        "X_test": run_dir / f"{run_name}_X_test_pca64_lda11.npy",
        "y_train": run_dir / f"{run_name}_y_train_3group.npy",
        "y_val": run_dir / f"{run_name}_y_val_3group.npy",
        "y_test": run_dir / f"{run_name}_y_test_3group.npy",
    }
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "The gamma-only QKA split must be generated first. Missing:\n" + "\n".join(missing)
        )
    arrays = {k: np.load(v) for k, v in paths.items()}
    return run_dir, run_name, arrays


def choose_by_validation(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    return max(candidates, key=lambda r: (r["val_macro_f1"], r["val_accuracy"]))


def save_common_outputs(out_dir: Path, run_name: str, y_test, y_pred, metrics: dict[str, Any]) -> None:
    np.save(out_dir / f"{run_name}_predictions.npy", y_pred)
    np.save(out_dir / f"{run_name}_confusion_matrix.npy", confusion_matrix(y_test, y_pred, labels=[0, 1, 2]))
    report = classification_report(
        y_test,
        y_pred,
        target_names=[GROUP_NAMES[i] for i in range(3)],
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(out_dir / f"{run_name}_classification_report.csv")
    pd.DataFrame([metrics]).to_csv(out_dir / f"{run_name}_results_summary.csv", index=False)
    with open(out_dir / f"{run_name}_results_summary.json", "w") as f:
        json.dump(metrics, f, indent=2)


def run_rbf(arrays: dict[str, np.ndarray], out_dir: Path, train_per_group: int, seed: int) -> dict[str, Any]:
    run_name = f"rbf_svm_3group_{train_per_group}_seed{seed}"
    X_train, X_val, X_test = arrays["X_train"], arrays["X_val"], arrays["X_test"]
    y_train, y_val, y_test = arrays["y_train"], arrays["y_val"], arrays["y_test"]

    candidates: list[dict[str, Any]] = []
    search_start = time.perf_counter()
    for C in SVC_C_GRID:
        for gamma in RBF_GAMMA_GRID:
            model = SVC(kernel="rbf", C=C, gamma=gamma, class_weight="balanced")
            model.fit(X_train, y_train)
            pred = model.predict(X_val)
            candidates.append({
                "C": C,
                "gamma": gamma,
                "val_macro_f1": f1_score(y_val, pred, average="macro", zero_division=0),
                "val_accuracy": accuracy_score(y_val, pred),
            })
    search_time = time.perf_counter() - search_start
    best = choose_by_validation(candidates)

    final_start = time.perf_counter()
    model = SVC(kernel="rbf", C=best["C"], gamma=best["gamma"], class_weight="balanced")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    final_time = time.perf_counter() - final_start

    metrics = {
        "model": "RBF-SVM",
        "train_per_group": train_per_group,
        "seed": seed,
        "train_seed": seed,
        "validation_seed": seed + 1,
        "test_seed": seed + 2,
        "num_train": int(len(y_train)),
        "num_val": int(len(y_val)),
        "num_test": int(len(y_test)),
        "selected_C": float(best["C"]),
        "selected_gamma": best["gamma"],
        "best_val_macro_f1": float(best["val_macro_f1"]),
        "best_val_accuracy": float(best["val_accuracy"]),
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "test_macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "test_weighted_f1": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "search_time_sec": search_time,
        "final_fit_predict_time_sec": final_time,
        "total_wall_time_sec": search_time + final_time,
        "representation": "SLR (PCA64->LDA11), unscaled classical input",
    }
    pd.DataFrame(candidates).to_csv(out_dir / f"{run_name}_validation_grid.csv", index=False)
    save_common_outputs(out_dir, run_name, y_test, y_pred, metrics)
    return metrics


def build_feature_map():
    from qiskit.circuit.library import ZZFeatureMap

    fm = ZZFeatureMap(
        feature_dimension=N_QUBITS,
        reps=FEATURE_MAP_REPS,
        entanglement=ENTANGLEMENT,
        parameter_prefix="x",
    )
    params = sorted(fm.parameters, key=lambda p: p.name)
    return fm, params


def get_backend(device: str, precision: str):
    from qiskit_aer import AerSimulator

    return AerSimulator(method="statevector", device=device, precision=precision)


def compute_statevectors(X: np.ndarray, feature_map, params, backend) -> np.ndarray:
    from qiskit import transpile

    circuits = []
    for i, x in enumerate(X):
        bind = {p: float(v) for p, v in zip(params, x)}
        circ = feature_map.assign_parameters(bind, inplace=False)
        circ.save_statevector()
        circ.name = f"fixed_sv_{i}"
        circuits.append(circ)
    transpiled = transpile(circuits, backend)
    result = backend.run(transpiled).result()
    states = [np.asarray(result.get_statevector(i), dtype=np.complex64) for i in range(len(circuits))]
    return np.vstack(states)


def qkernel(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return np.asarray(np.abs(A @ B.conj().T) ** 2, dtype=np.float32)


def train_only_scale_01(X_train, X_val, X_test):
    lo = X_train.min(axis=0)
    hi = X_train.max(axis=0)
    denom = np.maximum(hi - lo, 1e-12)
    scale = lambda X: np.clip((X - lo) / denom, 0.0, 1.0).astype(np.float32)
    return scale(X_train), scale(X_val), scale(X_test), lo, hi


def run_fixed_qk(
    arrays: dict[str, np.ndarray],
    out_dir: Path,
    train_per_group: int,
    seed: int,
    device: str,
    precision: str,
) -> dict[str, Any]:
    run_name = f"fixed_qk_3group_{train_per_group}_seed{seed}"
    X_train, X_val, X_test, lo, hi = train_only_scale_01(
        arrays["X_train"], arrays["X_val"], arrays["X_test"]
    )
    y_train, y_val, y_test = arrays["y_train"], arrays["y_val"], arrays["y_test"]
    np.save(out_dir / f"{run_name}_rescale_train_min.npy", lo)
    np.save(out_dir / f"{run_name}_rescale_train_max.npy", hi)

    fm, params = build_feature_map()
    backend = get_backend(device, precision)

    kernel_start = time.perf_counter()
    sv_train = compute_statevectors(X_train, fm, params, backend)
    sv_val = compute_statevectors(X_val, fm, params, backend)
    sv_test = compute_statevectors(X_test, fm, params, backend)
    K_train_raw = qkernel(sv_train, sv_train)
    K_train = K_train_raw + KERNEL_RIDGE * np.eye(len(y_train), dtype=np.float32)
    K_val_train = qkernel(sv_val, sv_train)
    K_test_train = qkernel(sv_test, sv_train)
    kernel_time = time.perf_counter() - kernel_start

    np.save(out_dir / f"{run_name}_kernel_train_raw.npy", K_train_raw)
    np.save(out_dir / f"{run_name}_kernel_test_train.npy", K_test_train)

    candidates: list[dict[str, Any]] = []
    for C in SVC_C_GRID:
        clf = SVC(kernel="precomputed", C=C, class_weight="balanced")
        clf.fit(K_train, y_train)
        pred = clf.predict(K_val_train)
        candidates.append({
            "C": C,
            "val_macro_f1": f1_score(y_val, pred, average="macro", zero_division=0),
            "val_accuracy": accuracy_score(y_val, pred),
        })
    best = choose_by_validation(candidates)

    classify_start = time.perf_counter()
    clf = SVC(kernel="precomputed", C=best["C"], class_weight="balanced")
    clf.fit(K_train, y_train)
    y_pred = clf.predict(K_test_train)
    classify_time = time.perf_counter() - classify_start

    metrics = {
        "model": "Fixed QK",
        "train_per_group": train_per_group,
        "seed": seed,
        "train_seed": seed,
        "validation_seed": seed + 1,
        "test_seed": seed + 2,
        "num_train": int(len(y_train)),
        "num_val": int(len(y_val)),
        "num_test": int(len(y_test)),
        "selected_C": float(best["C"]),
        "best_val_macro_f1": float(best["val_macro_f1"]),
        "best_val_accuracy": float(best["val_accuracy"]),
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "test_macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "test_weighted_f1": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "kernel_time_sec": kernel_time,
        "classifier_time_sec": classify_time,
        "total_wall_time_sec": kernel_time + classify_time,
        "kernel_ridge": KERNEL_RIDGE,
        "encoding_range": "[0,1]",
        "aer_device": device,
        "aer_precision": precision,
        "representation": "SLR (PCA64->LDA11), train-only min-max scaled",
    }
    pd.DataFrame(candidates).to_csv(out_dir / f"{run_name}_validation_grid.csv", index=False)
    save_common_outputs(out_dir, run_name, y_test, y_pred, metrics)
    return metrics


def main() -> None:
    args = parse_args()
    _, _, arrays = load_saved_split(args.gamma_results_root, args.train_per_group, args.seed)
    out_dir = args.results_root / f"train{args.train_per_group}_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "system_info.json", "w") as f:
        json.dump({
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "train_per_group": args.train_per_group,
            "seed": args.seed,
            "models": args.models,
            "gamma_results_root": str(args.gamma_results_root),
            "aer_device": args.aer_device,
            "precision": args.precision,
        }, f, indent=2)
    with open(out_dir / "nvidia_smi.txt", "w") as f:
        f.write(get_nvidia_smi_text())

    rows = []
    if "rbf" in args.models:
        target = out_dir / f"rbf_svm_3group_{args.train_per_group}_seed{args.seed}_results_summary.json"
        if not (args.skip_existing and target.exists()):
            print(f"Running RBF-SVM: train/group={args.train_per_group}, seed={args.seed}", flush=True)
            rows.append(run_rbf(arrays, out_dir, args.train_per_group, args.seed))
        else:
            print(f"Skipping existing {target}", flush=True)

    if "fixed_qk" in args.models:
        target = out_dir / f"fixed_qk_3group_{args.train_per_group}_seed{args.seed}_results_summary.json"
        if not (args.skip_existing and target.exists()):
            print(f"Running fixed QK: train/group={args.train_per_group}, seed={args.seed}", flush=True)
            rows.append(run_fixed_qk(
                arrays, out_dir, args.train_per_group, args.seed, args.aer_device, args.precision
            ))
        else:
            print(f"Skipping existing {target}", flush=True)

    if rows:
        pd.DataFrame(rows).to_csv(out_dir / "baseline_run_summary.csv", index=False)
        print(pd.DataFrame(rows).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
