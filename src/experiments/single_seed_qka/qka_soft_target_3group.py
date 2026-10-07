from __future__ import annotations

import json
import platform
import subprocess
import time
import warnings
from pathlib import Path
from typing import Tuple

import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
from src.paths import EMBEDDINGS_DIR, RESULTS_DIR

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.svm import SVC

warnings.filterwarnings("ignore")


'''
| Run          | Train per group | Total train | Val per group |     Test | Purpose                           |
| ------------ | --------------: | ----------: | ------------: | -------: | --------------------------------- |
| QKA-Train100 |             100 |         300 |            50 |      561 | Baseline                          |
| QKA-Train150 |             150 |         450 |            50 | same 561 | Check sample scaling              |
| QKA-Train210 |             210 |   up to 630 |            50 | same 561 | Maximum balanced training setting |

'''

# ============================================================
# 1. CONFIG
# ============================================================

SEED = 42
rng = np.random.default_rng(SEED)

# Folder from matched classical run. This should contain:
# train_embeddings_pca.npy
# valid_embeddings_pca.npy
# test_embeddings_pca.npy
# lda_model_11d.joblib


# Raw embedding folder containing full labels
MATCHED_DIR = RESULTS_DIR / "matched_representation"

RAW_LABEL_DIR = EMBEDDINGS_DIR / "efficientnet_b0"

RUN_RESULTS_DIR = (
    RESULTS_DIR
    / "single_seed_qka"
    / "soft_target_gamma_qka_210"
)

RUN_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RUN_NAME = "soft_target_gamma_scaled_qka_210"

N_QUBITS = 11

# Keep this moderate first. 210/group gives up to 630 train samples and is much heavier.
MAX_TRAIN_PER_GROUP = 210
MAX_VAL_PER_GROUP = 50
MAX_TEST_PER_GROUP = 210

FEATURE_MAP_REPS = 3
ANSATZ_REPS = 2
ENTANGLEMENT = "circular"

SPSA_MAXITER = 300
SPSA_A = 0.08
SPSA_C = 0.12
SPSA_ALPHA = 0.602
SPSA_GAMMA = 0.101
SPSA_A_OFFSET = 10.0
LOG_EVERY = 1
EVAL_VAL_EVERY = 5

SVC_C_GRID = [0.1, 1.0, 10.0, 100.0]

USE_AER = True
USE_AER_GPU = True
AER_PRECISION = "single"

KERNEL_RIDGE = 0.1

# ============================================================
# 2. CORRECT 12-CLASS TO 3-GROUP MAPPING
# ============================================================

# C0  = complex                         -> Multiple
# C1  = frog_eye_leaf_spot              -> Single
# C2  = frog_eye_leaf_spot complex      -> Multiple
# C3  = healthy                         -> Healthy
# C4  = powdery_mildew                  -> Single
# C5  = powdery_mildew complex          -> Multiple
# C6  = rust                            -> Single
# C7  = rust complex                    -> Multiple
# C8  = rust frog_eye_leaf_spot         -> Multiple
# C9  = scab                            -> Single
# C10 = scab frog_eye_leaf_spot         -> Multiple
# C11 = scab frog_eye_leaf_spot complex -> Multiple

LABEL12_TO_GROUP3 = {
    0: 2,
    1: 1,
    2: 2,
    3: 0,
    4: 1,
    5: 2,
    6: 1,
    7: 2,
    8: 2,
    9: 1,
    10: 2,
    11: 2,
}

GROUP_NAMES = {
    0: "Healthy",
    1: "Single",
    2: "Multiple",
}


def map_labels_to_3_groups(y_raw: np.ndarray) -> np.ndarray:
    return np.array([LABEL12_TO_GROUP3[int(y)] for y in y_raw], dtype=np.int64)


# ============================================================
# 3. SYSTEM LOGGING
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
    except Exception as e:
        return f"Could not run nvidia-smi: {repr(e)}"


def save_system_info() -> None:
    info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "seed": SEED,
        "matched_dir": str(MATCHED_DIR),
        "raw_label_dir": str(RAW_LABEL_DIR),
        "results_dir": str(RUN_RESULTS_DIR),
        "run_name": RUN_NAME,
        "task": "corrected 3-group Healthy/Single/Multiple QKA",
        "max_train_per_group": MAX_TRAIN_PER_GROUP,
        "max_val_per_group": MAX_VAL_PER_GROUP,
        "max_test_per_group": MAX_TEST_PER_GROUP,
        "n_qubits": N_QUBITS,
        "feature_map_reps": FEATURE_MAP_REPS,
        "ansatz_reps": ANSATZ_REPS,
        "entanglement": ENTANGLEMENT,
        "spsa_maxiter": SPSA_MAXITER,
        "kernel_ridge": KERNEL_RIDGE,
        "use_aer_gpu": USE_AER_GPU,
        "aer_precision": AER_PRECISION,
    }

    with open(RUN_RESULTS_DIR / f"{RUN_NAME}_system_info.json", "w") as f:
        json.dump(info, f, indent=2)

    with open(RUN_RESULTS_DIR / f"{RUN_NAME}_nvidia_smi.txt", "w") as f:
        f.write(get_nvidia_smi_text())


# ============================================================
# 4. PLOT STYLE
# ============================================================

PAPER_SEQ_CMAP = LinearSegmentedColormap.from_list(
    "paper_seq_green_blue",
    ["#f4f1e9", "#dbe8dc", "#a9c8bd", "#6f9fa0", "#355b7a"],
    N=256,
)

plt.rcParams.update({
    "font.size": 16,
    "axes.labelsize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


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
# 5. FULL DATA LOADING, LDA TRANSFORM, AND GROUP SAMPLING
# ============================================================

def sample_per_group(y: np.ndarray, max_per_group: int, seed: int) -> np.ndarray:
    rng_local = np.random.default_rng(seed)
    selected = []

    for group in [0, 1, 2]:
        idx = np.where(y == group)[0].copy()
        rng_local.shuffle(idx)
        selected.extend(idx[:min(max_per_group, len(idx))])

    selected = np.array(selected, dtype=np.int64)
    rng_local.shuffle(selected)
    return selected


def load_full_pca64_lda11_3group() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    required = [
        MATCHED_DIR / "train_embeddings_pca.npy",
        MATCHED_DIR / "valid_embeddings_pca.npy",
        MATCHED_DIR / "test_embeddings_pca.npy",
        MATCHED_DIR / "lda_model_11d.joblib",
        RAW_LABEL_DIR / "train_label_idxs.npy",
        RAW_LABEL_DIR / "valid_label_idxs.npy",
        RAW_LABEL_DIR / "test_label_idxs.npy",
    ]

    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required files:\n"
            + "\n".join(missing)
            + "\n\nYou need full PCA64 arrays and the saved LDA model, plus full split labels."
        )

    X_train_pca64_full = np.load(MATCHED_DIR / "train_embeddings_pca.npy")
    X_val_pca64_full = np.load(MATCHED_DIR / "valid_embeddings_pca.npy")
    X_test_pca64_full = np.load(MATCHED_DIR / "test_embeddings_pca.npy")

    y_train_12_full = np.load(RAW_LABEL_DIR / "train_label_idxs.npy")
    y_val_12_full = np.load(RAW_LABEL_DIR / "valid_label_idxs.npy")
    y_test_12_full = np.load(RAW_LABEL_DIR / "test_label_idxs.npy")

    lda = joblib.load(MATCHED_DIR / "lda_model_11d.joblib")

    X_train_lda_full = lda.transform(X_train_pca64_full)
    X_val_lda_full = lda.transform(X_val_pca64_full)
    X_test_lda_full = lda.transform(X_test_pca64_full)

    y_train_3_full = map_labels_to_3_groups(y_train_12_full)
    y_val_3_full = map_labels_to_3_groups(y_val_12_full)
    y_test_3_full = map_labels_to_3_groups(y_test_12_full)

    train_idx = sample_per_group(y_train_3_full, MAX_TRAIN_PER_GROUP, SEED)
    val_idx = sample_per_group(y_val_3_full, MAX_VAL_PER_GROUP, SEED + 1)
    test_idx = sample_per_group(y_test_3_full, MAX_TEST_PER_GROUP, SEED + 2)

    X_train = X_train_lda_full[train_idx]
    y_train = y_train_3_full[train_idx]

    X_val = X_val_lda_full[val_idx]
    y_val = y_val_3_full[val_idx]

    X_test = X_test_lda_full[test_idx]
    y_test = y_test_3_full[test_idx]

    print("Corrected 3-group QKA data:")
    print(f"  Train: X={X_train.shape}, y={y_train.shape}, counts={np.bincount(y_train, minlength=3)}")
    print(f"  Val  : X={X_val.shape}, y={y_val.shape}, counts={np.bincount(y_val, minlength=3)}")
    print(f"  Test : X={X_test.shape}, y={y_test.shape}, counts={np.bincount(y_test, minlength=3)}")

    np.save(RUN_RESULTS_DIR / f"{RUN_NAME}_train_indices_full.npy", train_idx)
    np.save(RUN_RESULTS_DIR / f"{RUN_NAME}_val_indices_full.npy", val_idx)
    np.save(RUN_RESULTS_DIR / f"{RUN_NAME}_test_indices_full.npy", test_idx)

    np.save(RUN_RESULTS_DIR / f"{RUN_NAME}_X_train_pca64_lda11.npy", X_train)
    np.save(RUN_RESULTS_DIR / f"{RUN_NAME}_X_val_pca64_lda11.npy", X_val)
    np.save(RUN_RESULTS_DIR / f"{RUN_NAME}_X_test_pca64_lda11.npy", X_test)

    np.save(RUN_RESULTS_DIR / f"{RUN_NAME}_y_train_3group.npy", y_train)
    np.save(RUN_RESULTS_DIR / f"{RUN_NAME}_y_val_3group.npy", y_val)
    np.save(RUN_RESULTS_DIR / f"{RUN_NAME}_y_test_3group.npy", y_test)

    return X_train, y_train, X_val, y_val, X_test, y_test


# ============================================================
# 6. RESCALING
# ============================================================

def rescale_train_only(X_train, X_val, X_test):
    train_min = X_train.min(axis=0)
    train_max = X_train.max(axis=0)
    denom = np.maximum(train_max - train_min, 1e-12)

    X_train_01 = np.clip((X_train - train_min) / denom, 0.0, 1.0)
    X_val_01 = np.clip((X_val - train_min) / denom, 0.0, 1.0)
    X_test_01 = np.clip((X_test - train_min) / denom, 0.0, 1.0)

    np.save(RUN_RESULTS_DIR / f"{RUN_NAME}_rescale_train_min.npy", train_min)
    np.save(RUN_RESULTS_DIR / f"{RUN_NAME}_rescale_train_max.npy", train_max)

    return X_train_01, X_val_01, X_test_01


# ============================================================
# 7. QISKIT CIRCUIT AND KERNEL
# ============================================================

def build_qka_circuit_objects():
    from qiskit.circuit.library import ZZFeatureMap

    feature_map = ZZFeatureMap(
        feature_dimension=N_QUBITS,
        reps=FEATURE_MAP_REPS,
        entanglement=ENTANGLEMENT,
        parameter_prefix="x",
    )

    circuit = feature_map
    feature_params = sorted(list(feature_map.parameters), key=lambda p: p.name)
    theta_params = []  # no ansatz

    print("3-group soft-target gamma-QKA circuit:")
    print(f"  Feature params: {len(feature_params)}")
    print(f"  Gamma params  : {N_QUBITS}")
    print(f"  Total params  : {N_QUBITS}")

    return circuit, feature_params, theta_params


def make_bound_circuit(circuit, feature_params, theta_params, x, combined_weights):
    gamma_w = combined_weights
    x_scaled = x * gamma_w

    bind_dict = {p: float(v) for p, v in zip(feature_params, x_scaled)}

    return circuit.assign_parameters(bind_dict, inplace=False)


def get_aer_backend():
    if not USE_AER:
        return None

    from qiskit_aer import AerSimulator

    backend = AerSimulator(
        method="statevector",
        device="GPU" if USE_AER_GPU else "CPU",
        precision=AER_PRECISION,
    )

    print(f"Using AerSimulator: GPU={USE_AER_GPU}, precision={AER_PRECISION}")
    return backend


def compute_statevectors(X, combined_weights, circuit, feature_params, theta_params, backend=None):
    if backend is not None:
        from qiskit import transpile

        circuits = []
        for idx, x in enumerate(X):
            bound = make_bound_circuit(circuit, feature_params, theta_params, x, combined_weights)
            bound.save_statevector()
            bound.name = f"sv_{idx}"
            circuits.append(bound)

        tcirc = transpile(circuits, backend)
        res = backend.run(tcirc).result()

        states = []
        for i in range(len(circuits)):
            states.append(np.asarray(res.get_statevector(i), dtype=np.complex64))

        return np.vstack(states)

    from qiskit.quantum_info import Statevector

    states = []
    for x in X:
        bound = make_bound_circuit(circuit, feature_params, theta_params, x, combined_weights)
        states.append(np.asarray(Statevector.from_instruction(bound).data, dtype=np.complex64))

    return np.vstack(states)


def quantum_kernel_from_states(states_a, states_b):
    return np.asarray(np.abs(states_a @ states_b.conj().T) ** 2, dtype=np.float32)


# ============================================================
# 8. SOFT TARGET AND OBJECTIVE
# ============================================================

def construct_soft_3group_target(y_train: np.ndarray) -> np.ndarray:
    """
    Soft pathology-aware target:
      same group = 1.0
      Healthy-Single = 0.2
      Healthy-Multiple = 0.0
      Single-Multiple = 0.5
    """
    sim = {
        (0, 0): 1.0,
        (1, 1): 1.0,
        (2, 2): 1.0,
        (0, 1): 0.2,
        (1, 0): 0.2,
        (0, 2): 0.0,
        (2, 0): 0.0,
        (1, 2): 0.5,
        (2, 1): 0.5,
    }

    n = len(y_train)
    T = np.zeros((n, n), dtype=np.float32)

    for i in range(n):
        for j in range(n):
            T[i, j] = sim[(int(y_train[i]), int(y_train[j]))]

    return T


def kernel_alignment(K, T):
    return float(np.sum(K * T) / (np.linalg.norm(K, ord="fro") * np.linalg.norm(T, ord="fro") + 1e-12))


class QKAObjective:
    def __init__(self, X_train, y_train, circuit, feature_params, theta_params, backend=None):
        self.X_train = X_train
        self.y_train = y_train
        self.circuit = circuit
        self.feature_params = feature_params
        self.theta_params = theta_params
        self.backend = backend
        self.T = construct_soft_3group_target(y_train)

    def evaluate(self, combined_weights):
        states = compute_statevectors(
            self.X_train,
            combined_weights,
            self.circuit,
            self.feature_params,
            self.theta_params,
            self.backend,
        )
        K = quantum_kernel_from_states(states, states)
        align = kernel_alignment(K, self.T)
        return -align, align


def evaluate_on_validation(weights, X_train, y_train, X_val, y_val, circuit, feature_params, theta_params, backend):
    sv_train = compute_statevectors(X_train, weights, circuit, feature_params, theta_params, backend)
    sv_val = compute_statevectors(X_val, weights, circuit, feature_params, theta_params, backend)

    K_train = quantum_kernel_from_states(sv_train, sv_train)
    K_train = K_train + KERNEL_RIDGE * np.eye(len(y_train), dtype=np.float32)

    K_val_train = quantum_kernel_from_states(sv_val, sv_train)

    best = {
        "val_macro_f1": -1.0,
        "val_accuracy": -1.0,
        "C": None,
    }

    for C in SVC_C_GRID:
        clf = SVC(kernel="precomputed", C=C, class_weight="balanced")
        clf.fit(K_train, y_train)
        y_pred = clf.predict(K_val_train)

        val_macro_f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)
        val_acc = accuracy_score(y_val, y_pred)

        if (val_macro_f1, val_acc) > (best["val_macro_f1"], best["val_accuracy"]):
            best = {
                "val_macro_f1": val_macro_f1,
                "val_accuracy": val_acc,
                "C": C,
            }

    return best


# ============================================================
# 9. SPSA
# ============================================================

def run_spsa(objective, n_params, X_train, y_train, X_val, y_val, circuit, feature_params, theta_params, backend):
    num_ansatz = len(theta_params)

    init_theta = rng.uniform(-0.05, 0.05, size=num_ansatz)
    init_gamma = rng.uniform(0.95, 1.05, size=N_QUBITS)
    weights = np.concatenate([init_theta, init_gamma]).astype(np.float64)

    best_weights = weights.copy()
    best_val_macro_f1 = -1.0
    best_val_acc = -1.0
    best_C = None
    best_iter = 0

    log_rows = []

    print("\nStarting corrected 3-group adaptive QKA SPSA...")
    start = time.perf_counter()

    for k in range(1, SPSA_MAXITER + 1):
        ak = SPSA_A / ((k + SPSA_A_OFFSET) ** SPSA_ALPHA)
        ck = SPSA_C / (k ** SPSA_GAMMA)

        delta = rng.choice([-1.0, 1.0], size=n_params)

        w_plus = weights + ck * delta
        w_minus = weights - ck * delta

        loss_plus, align_plus = objective.evaluate(w_plus)
        loss_minus, align_minus = objective.evaluate(w_minus)

        grad_est = ((loss_plus - loss_minus) / (2.0 * ck)) * delta
        weights = weights - ak * grad_est

        _, align_current = objective.evaluate(weights)

        row = {
            "iteration": k,
            "loss_plus": loss_plus,
            "loss_minus": loss_minus,
            "alignment_plus": align_plus,
            "alignment_minus": align_minus,
            "alignment_current": align_current,
            "gradient_norm": float(np.linalg.norm(grad_est)),
            "weights_norm": float(np.linalg.norm(weights)),
        }

        if k % LOG_EVERY == 0:
            print(
                f"Iter {k:04d}/{SPSA_MAXITER} | "
                f"Alignment={align_current:.6f} | "
                f"GradNorm={row['gradient_norm']:.4f}"
            )

        if (k % EVAL_VAL_EVERY == 0) or (k == 1) or (k == SPSA_MAXITER):
            val_result = evaluate_on_validation(
                weights,
                X_train,
                y_train,
                X_val,
                y_val,
                circuit,
                feature_params,
                theta_params,
                backend,
            )

            row.update({
                "val_macro_f1": val_result["val_macro_f1"],
                "val_accuracy": val_result["val_accuracy"],
                "val_C": val_result["C"],
            })

            print(
                f"Validation iter {k:04d}: "
                f"C={val_result['C']} | "
                f"val_acc={val_result['val_accuracy']:.4f} | "
                f"val_macro_f1={val_result['val_macro_f1']:.4f}"
            )

            if (val_result["val_macro_f1"], val_result["val_accuracy"]) > (best_val_macro_f1, best_val_acc):
                best_weights = weights.copy()
                best_val_macro_f1 = val_result["val_macro_f1"]
                best_val_acc = val_result["val_accuracy"]
                best_C = val_result["C"]
                best_iter = k

                np.save(RUN_RESULTS_DIR / f"{RUN_NAME}_best_weights.npy", best_weights)

                with open(RUN_RESULTS_DIR / f"{RUN_NAME}_best_checkpoint.json", "w") as f:
                    json.dump(
                        {
                            "best_iter": best_iter,
                            "best_val_macro_f1": best_val_macro_f1,
                            "best_val_accuracy": best_val_acc,
                            "best_C": best_C,
                        },
                        f,
                        indent=2,
                    )

        log_rows.append(row)

        pd.DataFrame(log_rows).to_csv(RUN_RESULTS_DIR / f"{RUN_NAME}_training_log.csv", index=False)
        np.save(RUN_RESULTS_DIR / f"{RUN_NAME}_latest_weights.npy", weights)

    total_time = time.perf_counter() - start

    print("\nSPSA complete.")
    print(f"Best validation checkpoint: iter={best_iter}, C={best_C}, val_macro_f1={best_val_macro_f1:.4f}")

    log_df = pd.DataFrame(log_rows)
    log_df["spsa_total_time_sec"] = total_time
    log_df.to_csv(RUN_RESULTS_DIR / f"{RUN_NAME}_training_log_final.csv", index=False)

    return best_weights, best_C


# ============================================================
# 10. MAIN
# ============================================================

def main():
    total_start = time.perf_counter()
    save_system_info()

    X_train, y_train, X_val, y_val, X_test, y_test = load_full_pca64_lda11_3group()

    X_train_01, X_val_01, X_test_01 = rescale_train_only(X_train, X_val, X_test)

    circuit, feature_params, theta_params = build_qka_circuit_objects()
    backend = get_aer_backend()

    total_params = len(theta_params) + N_QUBITS

    run_config = {
        "run_name": RUN_NAME,
        "method": "corrected adaptive soft-target 3-group QKA",
        "train_source": "full train split transformed using saved PCA64 and LDA11",
        "label_mapping": LABEL12_TO_GROUP3,
        "max_train_per_group": MAX_TRAIN_PER_GROUP,
        "max_val_per_group": MAX_VAL_PER_GROUP,
        "max_test_per_group": MAX_TEST_PER_GROUP,
        "num_train": int(len(y_train)),
        "num_val": int(len(y_val)),
        "num_test": int(len(y_test)),
        "n_qubits": N_QUBITS,
        "feature_map_reps": FEATURE_MAP_REPS,
        "ansatz_reps": ANSATZ_REPS,
        "entanglement": ENTANGLEMENT,
        "total_trainable_params": int(total_params),
        "soft_target": "same=1.0, Healthy-Single=0.2, Healthy-Multiple=0.0, Single-Multiple=0.5",
        "svc_c_grid": SVC_C_GRID,
    }

    with open(RUN_RESULTS_DIR / f"{RUN_NAME}_run_config.json", "w") as f:
        json.dump(run_config, f, indent=2)

    objective = QKAObjective(
        X_train_01,
        y_train,
        circuit,
        feature_params,
        theta_params,
        backend,
    )

    best_weights, best_C = run_spsa(
        objective,
        total_params,
        X_train_01,
        y_train,
        X_val_01,
        y_val,
        circuit,
        feature_params,
        theta_params,
        backend,
    )

    print("\nComputing final train/test kernels...")

    kernel_start = time.perf_counter()

    sv_train = compute_statevectors(X_train_01, best_weights, circuit, feature_params, theta_params, backend)
    sv_test = compute_statevectors(X_test_01, best_weights, circuit, feature_params, theta_params, backend)

    K_train = quantum_kernel_from_states(sv_train, sv_train)
    K_train = K_train + KERNEL_RIDGE * np.eye(len(y_train), dtype=np.float32)

    K_test_train = quantum_kernel_from_states(sv_test, sv_train)

    kernel_time = time.perf_counter() - kernel_start

    np.save(RUN_RESULTS_DIR / f"{RUN_NAME}_kernel_train_final.npy", K_train)
    np.save(RUN_RESULTS_DIR / f"{RUN_NAME}_kernel_test_train_final.npy", K_test_train)

    clf = SVC(kernel="precomputed", C=best_C, class_weight="balanced")
    clf.fit(K_train, y_train)
    y_pred = clf.predict(K_test_train)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

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

    np.save(RUN_RESULTS_DIR / f"{RUN_NAME}_predictions.npy", y_pred)

    pd.DataFrame(report_dict).transpose().to_csv(
        RUN_RESULTS_DIR / f"{RUN_NAME}_classification_report.csv"
    )

    with open(RUN_RESULTS_DIR / f"{RUN_NAME}_classification_report.txt", "w") as f:
        f.write(report_text)

    save_confusion_matrix_pretty(
        cm,
        RUN_RESULTS_DIR / f"{RUN_NAME}_confusion_matrix_counts_pretty_largefont_notitle",
    )

    metrics = {
        "test_accuracy": acc,
        "test_macro_f1": macro_f1,
        "test_weighted_f1": weighted_f1,
        "selected_C": best_C,
        "num_train": int(len(y_train)),
        "num_val": int(len(y_val)),
        "num_test": int(len(y_test)),
        "kernel_time_sec": kernel_time,
        "total_wall_time_sec": time.perf_counter() - total_start,
    }

    pd.DataFrame([metrics]).to_csv(
        RUN_RESULTS_DIR / f"{RUN_NAME}_results_summary.csv",
        index=False,
    )

    with open(RUN_RESULTS_DIR / f"{RUN_NAME}_results_summary.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "=" * 80)
    print("CORRECTED 3-GROUP QKA RESULT")
    print("=" * 80)
    for k, v in metrics.items():
        print(f"{k}: {v}")

    print(f"\nSaved to: {RUN_RESULTS_DIR}")


if __name__ == "__main__":
    main()
