from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
import warnings
from pathlib import Path

import sys
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.paths import EMBEDDINGS_DIR, RESULTS_DIR
from typing import Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

# ============================================================
# 1. DEFAULT CONFIGURATION
# ============================================================

SEED = 42
N_QUBITS = 11
FEATURE_MAP_REPS = 3
ENTANGLEMENT = "circular"

MAX_VAL_PER_GROUP = 50
MAX_TEST_PER_GROUP = 210

SPSA_MAXITER = 300
SPSA_A = 0.08
SPSA_C = 0.12
SPSA_ALPHA = 0.602
SPSA_GAMMA_EXPONENT = 0.101
SPSA_A_OFFSET = 10.0
LOG_EVERY = 1
EVAL_VAL_EVERY = 5

SVC_C_GRID = [0.1, 1.0, 10.0, 100.0]
KERNEL_RIDGE = 0.1

USE_AER = True
USE_AER_GPU = True
AER_PRECISION = "single"

MATCHED_DIR_DEFAULT = RESULTS_DIR / "matched_representation"
RAW_LABEL_DIR_DEFAULT = EMBEDDINGS_DIR / "efficientnet_b0"
RESULTS_ROOT_DEFAULT = RESULTS_DIR / "qka_gamma_only"

# ============================================================
# 2. LABEL MAPPING
# ============================================================

LABEL12_TO_GROUP3 = {
    0: 2,   # complex -> Multiple
    1: 1,   # frog_eye_leaf_spot -> Single
    2: 2,   # frog_eye_leaf_spot complex -> Multiple
    3: 0,   # healthy -> Healthy
    4: 1,   # powdery_mildew -> Single
    5: 2,   # powdery_mildew complex -> Multiple
    6: 1,   # rust -> Single
    7: 2,   # rust complex -> Multiple
    8: 2,   # rust frog_eye_leaf_spot -> Multiple
    9: 1,   # scab -> Single
    10: 2,  # scab frog_eye_leaf_spot -> Multiple
    11: 2,  # scab frog_eye_leaf_spot complex -> Multiple
}

GROUP_NAMES = {0: "Healthy", 1: "Single", 2: "Multiple"}


def map_labels_to_3_groups(y_raw: np.ndarray) -> np.ndarray:
    return np.asarray([LABEL12_TO_GROUP3[int(y)] for y in y_raw], dtype=np.int64)


# ============================================================
# 3. ARGUMENTS AND RUN PATHS
# ============================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Gamma-only soft-target quantum-kernel alignment for the matched "
            "three-group Plant Pathology experiment."
        )
    )
    parser.add_argument(
        "--train-per-group",
        type=int,
        choices=(100, 150, 210),
        default=210,
        help="Maximum number of training samples per macro group.",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--matched-dir", type=Path, default=MATCHED_DIR_DEFAULT)
    parser.add_argument("--raw-label-dir", type=Path, default=RAW_LABEL_DIR_DEFAULT)
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT_DEFAULT)
    parser.add_argument(
        "--aer-device",
        choices=("GPU", "CPU"),
        default="GPU" if USE_AER_GPU else "CPU",
    )
    parser.add_argument(
        "--precision",
        choices=("single", "double"),
        default=AER_PRECISION,
    )
    return parser.parse_args()


# ============================================================
# 4. SYSTEM LOGGING
# ============================================================


def get_nvidia_smi_text() -> str:
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout
        return f"nvidia-smi returned code {result.returncode}\nSTDERR:\n{result.stderr}"
    except Exception as exc:
        return f"Could not run nvidia-smi: {exc!r}"


# ============================================================
# 5. PLOT STYLE
# ============================================================

PAPER_SEQ_CMAP = LinearSegmentedColormap.from_list(
    "paper_seq_green_blue",
    ["#f4f1e9", "#dbe8dc", "#a9c8bd", "#6f9fa0", "#355b7a"],
    N=256,
)

plt.rcParams.update(
    {
        "font.size": 16,
        "axes.labelsize": 18,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def save_confusion_matrix_pretty(cm: np.ndarray, out_base: Path) -> None:
    labels = [GROUP_NAMES[i] for i in range(3)]
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap=PAPER_SEQ_CMAP,
        cbar=True,
        square=True,
        linewidths=0.5,
        linecolor="white",
        xticklabels=labels,
        yticklabels=labels,
        annot_kws={"size": 14},
        ax=ax,
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticklabels(labels, rotation=0)
    ax.set_yticklabels(labels, rotation=90, va="center")
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 6. DATA LOADING AND SAMPLING
# ============================================================


def sample_per_group(y: np.ndarray, max_per_group: int, seed: int) -> np.ndarray:
    rng_local = np.random.default_rng(seed)
    selected: list[int] = []
    for group in (0, 1, 2):
        idx = np.where(y == group)[0].copy()
        rng_local.shuffle(idx)
        selected.extend(idx[: min(max_per_group, len(idx))].tolist())
    selected_array = np.asarray(selected, dtype=np.int64)
    rng_local.shuffle(selected_array)
    return selected_array


def load_full_pca64_lda11_3group(
    matched_dir: Path,
    raw_label_dir: Path,
    results_dir: Path,
    run_name: str,
    train_per_group: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    required = [
        matched_dir / "train_embeddings_pca.npy",
        matched_dir / "valid_embeddings_pca.npy",
        matched_dir / "test_embeddings_pca.npy",
        matched_dir / "lda_model_11d.joblib",
        raw_label_dir / "train_label_idxs.npy",
        raw_label_dir / "valid_label_idxs.npy",
        raw_label_dir / "test_label_idxs.npy",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))

    X_train_pca64 = np.load(matched_dir / "train_embeddings_pca.npy")
    X_val_pca64 = np.load(matched_dir / "valid_embeddings_pca.npy")
    X_test_pca64 = np.load(matched_dir / "test_embeddings_pca.npy")

    y_train_12 = np.load(raw_label_dir / "train_label_idxs.npy")
    y_val_12 = np.load(raw_label_dir / "valid_label_idxs.npy")
    y_test_12 = np.load(raw_label_dir / "test_label_idxs.npy")

    lda = joblib.load(matched_dir / "lda_model_11d.joblib")
    X_train_lda = lda.transform(X_train_pca64)
    X_val_lda = lda.transform(X_val_pca64)
    X_test_lda = lda.transform(X_test_pca64)

    y_train_3 = map_labels_to_3_groups(y_train_12)
    y_val_3 = map_labels_to_3_groups(y_val_12)
    y_test_3 = map_labels_to_3_groups(y_test_12)

    train_idx = sample_per_group(y_train_3, train_per_group, seed)
    val_idx = sample_per_group(y_val_3, MAX_VAL_PER_GROUP, seed + 1)
    test_idx = sample_per_group(y_test_3, MAX_TEST_PER_GROUP, seed + 2)

    X_train, y_train = X_train_lda[train_idx], y_train_3[train_idx]
    X_val, y_val = X_val_lda[val_idx], y_val_3[val_idx]
    X_test, y_test = X_test_lda[test_idx], y_test_3[test_idx]

    print("Gamma-only 3-group QKA data:")
    print(f"  Train: X={X_train.shape}, counts={np.bincount(y_train, minlength=3)}")
    print(f"  Val  : X={X_val.shape}, counts={np.bincount(y_val, minlength=3)}")
    print(f"  Test : X={X_test.shape}, counts={np.bincount(y_test, minlength=3)}")

    np.save(results_dir / f"{run_name}_train_indices_full.npy", train_idx)
    np.save(results_dir / f"{run_name}_val_indices_full.npy", val_idx)
    np.save(results_dir / f"{run_name}_test_indices_full.npy", test_idx)
    np.save(results_dir / f"{run_name}_X_train_pca64_lda11.npy", X_train)
    np.save(results_dir / f"{run_name}_X_val_pca64_lda11.npy", X_val)
    np.save(results_dir / f"{run_name}_X_test_pca64_lda11.npy", X_test)
    np.save(results_dir / f"{run_name}_y_train_3group.npy", y_train)
    np.save(results_dir / f"{run_name}_y_val_3group.npy", y_val)
    np.save(results_dir / f"{run_name}_y_test_3group.npy", y_test)

    return X_train, y_train, X_val, y_val, X_test, y_test


# ============================================================
# 7. TRAIN-ONLY RESCALING
# ============================================================


def rescale_train_only(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    results_dir: Path,
    run_name: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_min = X_train.min(axis=0)
    train_max = X_train.max(axis=0)
    denom = np.maximum(train_max - train_min, 1e-12)

    X_train_01 = np.clip((X_train - train_min) / denom, 0.0, 1.0)
    X_val_01 = np.clip((X_val - train_min) / denom, 0.0, 1.0)
    X_test_01 = np.clip((X_test - train_min) / denom, 0.0, 1.0)

    np.save(results_dir / f"{run_name}_rescale_train_min.npy", train_min)
    np.save(results_dir / f"{run_name}_rescale_train_max.npy", train_max)
    return X_train_01, X_val_01, X_test_01


# ============================================================
# 8. GAMMA-ONLY QUANTUM FEATURE MAP AND KERNEL
# ============================================================


def build_feature_map():
    from qiskit.circuit.library import ZZFeatureMap

    feature_map = ZZFeatureMap(
        feature_dimension=N_QUBITS,
        reps=FEATURE_MAP_REPS,
        entanglement=ENTANGLEMENT,
        parameter_prefix="x",
    )
    feature_params = sorted(feature_map.parameters, key=lambda p: p.name)
    if len(feature_params) != N_QUBITS:
        raise RuntimeError(
            f"Expected {N_QUBITS} feature parameters, found {len(feature_params)}."
        )

    print("Gamma-only adaptive QKA feature map:")
    print(f"  Feature parameters : {len(feature_params)}")
    print(f"  Trainable gamma    : {N_QUBITS}")
    print("  Post-encoding ansatz: none")
    return feature_map, feature_params


def make_bound_circuit(feature_map, feature_params, x: np.ndarray, gamma: np.ndarray):
    if gamma.shape != (N_QUBITS,):
        raise ValueError(f"gamma must have shape ({N_QUBITS},), got {gamma.shape}")
    x_scaled = x * gamma
    bind_dict = {parameter: float(value) for parameter, value in zip(feature_params, x_scaled)}
    return feature_map.assign_parameters(bind_dict, inplace=False)


def get_aer_backend(device: str, precision: str):
    if not USE_AER:
        return None
    from qiskit_aer import AerSimulator

    backend = AerSimulator(method="statevector", device=device, precision=precision)
    print(f"Using AerSimulator: device={device}, precision={precision}")
    return backend


def compute_statevectors(
    X: np.ndarray,
    gamma: np.ndarray,
    feature_map,
    feature_params,
    backend=None,
) -> np.ndarray:
    if backend is not None:
        from qiskit import transpile

        circuits = []
        for idx, x in enumerate(X):
            bound = make_bound_circuit(feature_map, feature_params, x, gamma)
            bound.save_statevector()
            bound.name = f"sv_{idx}"
            circuits.append(bound)

        transpiled = transpile(circuits, backend)
        result = backend.run(transpiled).result()
        states = [np.asarray(result.get_statevector(i), dtype=np.complex64) for i in range(len(circuits))]
        return np.vstack(states)

    from qiskit.quantum_info import Statevector

    states = [
        np.asarray(
            Statevector.from_instruction(
                make_bound_circuit(feature_map, feature_params, x, gamma)
            ).data,
            dtype=np.complex64,
        )
        for x in X
    ]
    return np.vstack(states)


def quantum_kernel_from_states(states_a: np.ndarray, states_b: np.ndarray) -> np.ndarray:
    return np.asarray(np.abs(states_a @ states_b.conj().T) ** 2, dtype=np.float32)


# ============================================================
# 9. SOFT TARGET AND ALIGNMENT OBJECTIVE
# ============================================================


def construct_soft_3group_target(y_train: np.ndarray) -> np.ndarray:
    similarities = np.asarray(
        [
            [1.0, 0.2, 0.0],
            [0.2, 1.0, 0.5],
            [0.0, 0.5, 1.0],
        ],
        dtype=np.float32,
    )
    return similarities[y_train[:, None], y_train[None, :]]


def kernel_alignment(K: np.ndarray, T: np.ndarray) -> float:
    numerator = np.sum(K * T)
    denominator = np.linalg.norm(K, ord="fro") * np.linalg.norm(T, ord="fro") + 1e-12
    return float(numerator / denominator)


class GammaOnlyQKAObjective:
    def __init__(self, X_train, y_train, feature_map, feature_params, backend=None):
        self.X_train = X_train
        self.feature_map = feature_map
        self.feature_params = feature_params
        self.backend = backend
        self.target = construct_soft_3group_target(y_train)
        self.num_evaluations = 0

    def evaluate(self, gamma: np.ndarray) -> Tuple[float, float]:
        self.num_evaluations += 1
        states = compute_statevectors(
            self.X_train,
            gamma,
            self.feature_map,
            self.feature_params,
            self.backend,
        )
        kernel = quantum_kernel_from_states(states, states)
        alignment = kernel_alignment(kernel, self.target)
        return -alignment, alignment


def evaluate_on_validation(
    gamma,
    X_train,
    y_train,
    X_val,
    y_val,
    feature_map,
    feature_params,
    backend,
):
    sv_train = compute_statevectors(X_train, gamma, feature_map, feature_params, backend)
    sv_val = compute_statevectors(X_val, gamma, feature_map, feature_params, backend)

    K_train = quantum_kernel_from_states(sv_train, sv_train)
    K_train = K_train + KERNEL_RIDGE * np.eye(len(y_train), dtype=np.float32)
    K_val_train = quantum_kernel_from_states(sv_val, sv_train)

    best = {"val_macro_f1": -1.0, "val_accuracy": -1.0, "C": None}
    for C in SVC_C_GRID:
        classifier = SVC(kernel="precomputed", C=C, class_weight="balanced")
        classifier.fit(K_train, y_train)
        y_pred = classifier.predict(K_val_train)
        macro_f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)
        accuracy = accuracy_score(y_val, y_pred)
        if (macro_f1, accuracy) > (best["val_macro_f1"], best["val_accuracy"]):
            best = {"val_macro_f1": macro_f1, "val_accuracy": accuracy, "C": C}
    return best


# ============================================================
# 10. GAMMA-ONLY SPSA
# ============================================================


def run_spsa(
    objective,
    rng,
    X_train,
    y_train,
    X_val,
    y_val,
    feature_map,
    feature_params,
    backend,
    results_dir: Path,
    run_name: str,
):
    gamma = rng.uniform(0.95, 1.05, size=N_QUBITS).astype(np.float64)
    initial_gamma = gamma.copy()

    best_gamma = gamma.copy()
    best_val_macro_f1 = -1.0
    best_val_accuracy = -1.0
    best_C = None
    best_iteration = 0
    log_rows = []

    np.save(results_dir / f"{run_name}_initial_gamma.npy", initial_gamma)
    print("\nStarting gamma-only QKA SPSA...")
    start_time = time.perf_counter()

    for iteration in range(1, SPSA_MAXITER + 1):
        ak = SPSA_A / ((iteration + SPSA_A_OFFSET) ** SPSA_ALPHA)
        ck = SPSA_C / (iteration ** SPSA_GAMMA_EXPONENT)
        delta = rng.choice([-1.0, 1.0], size=N_QUBITS)

        gamma_plus = gamma + ck * delta
        gamma_minus = gamma - ck * delta

        loss_plus, alignment_plus = objective.evaluate(gamma_plus)
        loss_minus, alignment_minus = objective.evaluate(gamma_minus)

        gradient = ((loss_plus - loss_minus) / (2.0 * ck)) * delta
        gamma = gamma - ak * gradient
        _, alignment_current = objective.evaluate(gamma)

        row = {
            "iteration": iteration,
            "loss_plus": loss_plus,
            "loss_minus": loss_minus,
            "alignment_plus": alignment_plus,
            "alignment_minus": alignment_minus,
            "alignment_current": alignment_current,
            "gradient_norm": float(np.linalg.norm(gradient)),
            "gamma_norm": float(np.linalg.norm(gamma)),
            "gamma_min": float(np.min(gamma)),
            "gamma_max": float(np.max(gamma)),
            "objective_evaluations": objective.num_evaluations,
        }

        if iteration % LOG_EVERY == 0:
            print(
                f"Iter {iteration:04d}/{SPSA_MAXITER} | "
                f"alignment={alignment_current:.6f} | "
                f"grad_norm={row['gradient_norm']:.4f}"
            )

        if (
            iteration == 1
            or iteration % EVAL_VAL_EVERY == 0
            or iteration == SPSA_MAXITER
        ):
            validation = evaluate_on_validation(
                gamma,
                X_train,
                y_train,
                X_val,
                y_val,
                feature_map,
                feature_params,
                backend,
            )
            row.update(
                {
                    "val_macro_f1": validation["val_macro_f1"],
                    "val_accuracy": validation["val_accuracy"],
                    "val_C": validation["C"],
                }
            )
            print(
                f"Validation iter {iteration:04d}: C={validation['C']} | "
                f"val_acc={validation['val_accuracy']:.4f} | "
                f"val_macro_f1={validation['val_macro_f1']:.4f}"
            )

            if (validation["val_macro_f1"], validation["val_accuracy"]) > (
                best_val_macro_f1,
                best_val_accuracy,
            ):
                best_gamma = gamma.copy()
                best_val_macro_f1 = validation["val_macro_f1"]
                best_val_accuracy = validation["val_accuracy"]
                best_C = validation["C"]
                best_iteration = iteration
                np.save(results_dir / f"{run_name}_best_gamma.npy", best_gamma)
                with open(results_dir / f"{run_name}_best_checkpoint.json", "w") as handle:
                    json.dump(
                        {
                            "best_iteration": best_iteration,
                            "best_val_macro_f1": best_val_macro_f1,
                            "best_val_accuracy": best_val_accuracy,
                            "best_C": best_C,
                            "objective_evaluations": objective.num_evaluations,
                        },
                        handle,
                        indent=2,
                    )

        log_rows.append(row)
        pd.DataFrame(log_rows).to_csv(
            results_dir / f"{run_name}_training_log.csv", index=False
        )
        np.save(results_dir / f"{run_name}_latest_gamma.npy", gamma)

    elapsed = time.perf_counter() - start_time
    log_df = pd.DataFrame(log_rows)
    log_df["spsa_total_time_sec"] = elapsed
    log_df.to_csv(results_dir / f"{run_name}_training_log_final.csv", index=False)

    print("\nSPSA complete.")
    print(
        f"Best checkpoint: iter={best_iteration}, C={best_C}, "
        f"val_macro_f1={best_val_macro_f1:.4f}"
    )
    return best_gamma, best_C, best_iteration, elapsed


# ============================================================
# 11. MAIN
# ============================================================


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    run_name = f"qka_gamma_only_soft_3group_{args.train_per_group}_seed{args.seed}"
    results_dir = args.results_root / run_name
    results_dir.mkdir(parents=True, exist_ok=True)

    total_start = time.perf_counter()

    system_info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "seed": args.seed,
        "matched_dir": str(args.matched_dir),
        "raw_label_dir": str(args.raw_label_dir),
        "results_dir": str(results_dir),
        "run_name": run_name,
        "method": "gamma-only soft-target 3-group QKA",
        "train_per_group": args.train_per_group,
        "max_val_per_group": MAX_VAL_PER_GROUP,
        "max_test_per_group": MAX_TEST_PER_GROUP,
        "n_qubits": N_QUBITS,
        "feature_map_reps": FEATURE_MAP_REPS,
        "entanglement": ENTANGLEMENT,
        "spsa_maxiter": SPSA_MAXITER,
        "kernel_ridge": KERNEL_RIDGE,
        "aer_device": args.aer_device,
        "aer_precision": args.precision,
    }
    with open(results_dir / f"{run_name}_system_info.json", "w") as handle:
        json.dump(system_info, handle, indent=2)
    with open(results_dir / f"{run_name}_nvidia_smi.txt", "w") as handle:
        handle.write(get_nvidia_smi_text())

    X_train, y_train, X_val, y_val, X_test, y_test = load_full_pca64_lda11_3group(
        args.matched_dir,
        args.raw_label_dir,
        results_dir,
        run_name,
        args.train_per_group,
        args.seed,
    )
    X_train, X_val, X_test = rescale_train_only(
        X_train, X_val, X_test, results_dir, run_name
    )

    feature_map, feature_params = build_feature_map()
    backend = get_aer_backend(args.aer_device, args.precision)

    run_config = {
        **system_info,
        "num_train": int(len(y_train)),
        "num_val": int(len(y_val)),
        "num_test": int(len(y_test)),
        "total_trainable_params": N_QUBITS,
        "trainable_parameters": "gamma only",
        "gamma_initialization": "Uniform(0.95, 1.05)",
        "soft_target": (
            "same=1.0, Healthy-Single=0.2, Healthy-Multiple=0.0, "
            "Single-Multiple=0.5"
        ),
        "svc_c_grid": SVC_C_GRID,
        "spsa_a": SPSA_A,
        "spsa_c": SPSA_C,
        "spsa_alpha": SPSA_ALPHA,
        "spsa_gamma_exponent": SPSA_GAMMA_EXPONENT,
        "spsa_A_offset": SPSA_A_OFFSET,
    }
    with open(results_dir / f"{run_name}_run_config.json", "w") as handle:
        json.dump(run_config, handle, indent=2)

    objective = GammaOnlyQKAObjective(
        X_train, y_train, feature_map, feature_params, backend
    )
    best_gamma, best_C, best_iteration, spsa_time = run_spsa(
        objective,
        rng,
        X_train,
        y_train,
        X_val,
        y_val,
        feature_map,
        feature_params,
        backend,
        results_dir,
        run_name,
    )

    print("\nComputing final train/test kernels...")
    kernel_start = time.perf_counter()
    sv_train = compute_statevectors(
        X_train, best_gamma, feature_map, feature_params, backend
    )
    sv_test = compute_statevectors(
        X_test, best_gamma, feature_map, feature_params, backend
    )
    K_train_raw = quantum_kernel_from_states(sv_train, sv_train)
    K_train = K_train_raw + KERNEL_RIDGE * np.eye(len(y_train), dtype=np.float32)
    K_test_train = quantum_kernel_from_states(sv_test, sv_train)
    kernel_time = time.perf_counter() - kernel_start

    np.save(results_dir / f"{run_name}_kernel_train_raw_final.npy", K_train_raw)
    np.save(results_dir / f"{run_name}_kernel_train_regularized_final.npy", K_train)
    np.save(results_dir / f"{run_name}_kernel_test_train_final.npy", K_test_train)

    classifier = SVC(kernel="precomputed", C=best_C, class_weight="balanced")
    classifier.fit(K_train, y_train)
    y_pred = classifier.predict(K_test_train)

    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    final_alignment = kernel_alignment(
        K_train_raw, construct_soft_3group_target(y_train)
    )

    report_dict = classification_report(
        y_test,
        y_pred,
        target_names=[GROUP_NAMES[i] for i in range(3)],
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        y_test,
        y_pred,
        target_names=[GROUP_NAMES[i] for i in range(3)],
        zero_division=0,
    )
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])

    np.save(results_dir / f"{run_name}_predictions.npy", y_pred)
    pd.DataFrame(report_dict).transpose().to_csv(
        results_dir / f"{run_name}_classification_report.csv"
    )
    with open(results_dir / f"{run_name}_classification_report.txt", "w") as handle:
        handle.write(report_text)
    save_confusion_matrix_pretty(
        cm,
        results_dir / f"{run_name}_confusion_matrix_counts_pretty_largefont_notitle",
    )

    total_wall_time = time.perf_counter() - total_start
    metrics = {
        "test_accuracy": accuracy,
        "test_macro_f1": macro_f1,
        "test_weighted_f1": weighted_f1,
        "selected_C": best_C,
        "best_iteration": best_iteration,
        "final_training_alignment": final_alignment,
        "num_train": int(len(y_train)),
        "num_val": int(len(y_val)),
        "num_test": int(len(y_test)),
        "trainable_parameter_count": N_QUBITS,
        "objective_evaluations": objective.num_evaluations,
        "spsa_time_sec": spsa_time,
        "final_kernel_time_sec": kernel_time,
        "total_wall_time_sec": total_wall_time,
    }
    pd.DataFrame([metrics]).to_csv(
        results_dir / f"{run_name}_results_summary.csv", index=False
    )
    with open(results_dir / f"{run_name}_results_summary.json", "w") as handle:
        json.dump(metrics, handle, indent=2)

    print("\n" + "=" * 80)
    print("GAMMA-ONLY SOFT-TARGET 3-GROUP QKA RESULT")
    print("=" * 80)
    for key, value in metrics.items():
        print(f"{key}: {value}")
    print(f"\nSaved to: {results_dir}")


if __name__ == "__main__":
    main()
