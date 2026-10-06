from __future__ import annotations

import json
import platform
import subprocess
import time
import warnings
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

import sys
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from src.paths import RESULTS_DIR

warnings.filterwarnings("ignore")


# ============================================================
# 1. CONFIG
# ============================================================

SEED = 42

SPLIT_DIR = RESULTS_DIR / "fixed_qk_12class_50_per_class"

RUN_NAME = "matched_classical_12class_50_per_class"
RUN_RESULTS_DIR = RESULTS_DIR / "matched_classical_12class_50" / RUN_NAME
RUN_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

N_CLASSES = 12

CLASS_NAMES = {
    0: "complex",
    1: "frog_eye_leaf_spot",
    2: "frog_eye_leaf_spot_complex",
    3: "healthy",
    4: "powdery_mildew",
    5: "powdery_mildew_complex",
    6: "rust",
    7: "rust_complex",
    8: "rust_frog_eye_leaf_spot",
    9: "scab",
    10: "scab_frog_eye_leaf_spot",
    11: "scab_frog_eye_leaf_spot_complex",
}

# Small validation set, so keep grid compact.
LINEAR_C_GRID = [0.01, 0.1, 1.0, 10.0]
RBF_C_GRID = [0.1, 1.0, 10.0, 100.0]
RBF_GAMMA_GRID = ["scale", 0.01, 0.1, 1.0]

RF_N_ESTIMATORS_GRID = [200, 500]
RF_MAX_DEPTH_GRID = [None, 10, 20]
RF_MIN_SAMPLES_LEAF_GRID = [1, 2, 5]

XGB_N_ESTIMATORS_GRID = [200, 500]
XGB_MAX_DEPTH_GRID = [3, 5]
XGB_LEARNING_RATE_GRID = [0.03, 0.1]
XGB_SUBSAMPLE_GRID = [0.8, 1.0]


# ============================================================
# 2. SYSTEM INFO
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
        "run_name": RUN_NAME,
        "task": "matched classical baselines on 12-class 50/class fixed-QK split",
        "split_dir": str(SPLIT_DIR),
        "results_dir": str(RUN_RESULTS_DIR),
        "representation": "PCA64 -> LDA11",
        "train_source": "same saved split as fixed_qk_12class_50_per_class",
        "models": ["Linear SVM", "RBF-SVM", "Random Forest", "XGBoost"],
    }

    with open(RUN_RESULTS_DIR / f"{RUN_NAME}_system_info.json", "w") as f:
        json.dump(info, f, indent=2)

    with open(RUN_RESULTS_DIR / f"{RUN_NAME}_nvidia_smi.txt", "w") as f:
        f.write(get_nvidia_smi_text())


# ============================================================
# 3. DATA LOADING
# ============================================================

def load_saved_split():
    required = [
        SPLIT_DIR / "fixed_qk_12class_50_per_class_X_train_pca64_lda11.npy",
        SPLIT_DIR / "fixed_qk_12class_50_per_class_X_val_pca64_lda11.npy",
        SPLIT_DIR / "fixed_qk_12class_50_per_class_X_test_pca64_lda11.npy",
        SPLIT_DIR / "fixed_qk_12class_50_per_class_y_train_12class.npy",
        SPLIT_DIR / "fixed_qk_12class_50_per_class_y_val_12class.npy",
        SPLIT_DIR / "fixed_qk_12class_50_per_class_y_test_12class.npy",
    ]

    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required split files:\n" + "\n".join(missing)
        )

    X_train = np.load(SPLIT_DIR / "fixed_qk_12class_50_per_class_X_train_pca64_lda11.npy")
    X_val = np.load(SPLIT_DIR / "fixed_qk_12class_50_per_class_X_val_pca64_lda11.npy")
    X_test = np.load(SPLIT_DIR / "fixed_qk_12class_50_per_class_X_test_pca64_lda11.npy")

    y_train = np.load(SPLIT_DIR / "fixed_qk_12class_50_per_class_y_train_12class.npy").astype(np.int64)
    y_val = np.load(SPLIT_DIR / "fixed_qk_12class_50_per_class_y_val_12class.npy").astype(np.int64)
    y_test = np.load(SPLIT_DIR / "fixed_qk_12class_50_per_class_y_test_12class.npy").astype(np.int64)

    print("Loaded matched 12-class split:")
    print(f"  Train: X={X_train.shape}, y={y_train.shape}, counts={np.bincount(y_train, minlength=N_CLASSES)}")
    print(f"  Val  : X={X_val.shape}, y={y_val.shape}, counts={np.bincount(y_val, minlength=N_CLASSES)}")
    print(f"  Test : X={X_test.shape}, y={y_test.shape}, counts={np.bincount(y_test, minlength=N_CLASSES)}")

    split_summary = {
        "train_shape": list(X_train.shape),
        "val_shape": list(X_val.shape),
        "test_shape": list(X_test.shape),
        "train_counts": np.bincount(y_train, minlength=N_CLASSES).tolist(),
        "val_counts": np.bincount(y_val, minlength=N_CLASSES).tolist(),
        "test_counts": np.bincount(y_test, minlength=N_CLASSES).tolist(),
    }

    with open(RUN_RESULTS_DIR / f"{RUN_NAME}_split_summary.json", "w") as f:
        json.dump(split_summary, f, indent=2)

    pd.DataFrame({
        "class_id": list(range(N_CLASSES)),
        "class_name": [CLASS_NAMES[i] for i in range(N_CLASSES)],
        "train_count": np.bincount(y_train, minlength=N_CLASSES),
        "val_count": np.bincount(y_val, minlength=N_CLASSES),
        "test_count": np.bincount(y_test, minlength=N_CLASSES),
    }).to_csv(
        RUN_RESULTS_DIR / f"{RUN_NAME}_split_counts.csv",
        index=False,
    )

    return X_train, y_train, X_val, y_val, X_test, y_test


# ============================================================
# 4. EVALUATION HELPERS
# ============================================================

def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def save_confusion_matrix(cm: np.ndarray, out_base: Path) -> None:
    tick_labels = [f"C{i}" for i in range(N_CLASSES)]

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(np.arange(N_CLASSES))
    ax.set_yticks(np.arange(N_CLASSES))
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    ax.set_yticklabels(tick_labels)

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                fontsize=8,
            )

    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_model_outputs(
    model_name_safe: str,
    y_test: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    report_text = classification_report(
        y_test,
        y_pred,
        labels=list(range(N_CLASSES)),
        target_names=[CLASS_NAMES[i] for i in range(N_CLASSES)],
        zero_division=0,
    )

    report_dict = classification_report(
        y_test,
        y_pred,
        labels=list(range(N_CLASSES)),
        target_names=[CLASS_NAMES[i] for i in range(N_CLASSES)],
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(y_test, y_pred, labels=list(range(N_CLASSES)))

    np.save(RUN_RESULTS_DIR / f"{RUN_NAME}_{model_name_safe}_predictions.npy", y_pred)

    with open(RUN_RESULTS_DIR / f"{RUN_NAME}_{model_name_safe}_classification_report.txt", "w") as f:
        f.write(report_text)

    pd.DataFrame(report_dict).transpose().to_csv(
        RUN_RESULTS_DIR / f"{RUN_NAME}_{model_name_safe}_classification_report.csv"
    )

    save_confusion_matrix(
        cm,
        RUN_RESULTS_DIR / f"{RUN_NAME}_{model_name_safe}_confusion_matrix_counts",
    )


# ============================================================
# 5. MODEL TRAINING AND SELECTION
# ============================================================

def fit_select_test_model(
    model_name: str,
    model_name_safe: str,
    candidate_models: List[Dict[str, Any]],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, Any]:

    print("\n" + "=" * 80)
    print(f"Running {model_name}")
    print("=" * 80)

    rows = []
    best = None

    search_start = time.perf_counter()

    for cand_id, cand in enumerate(candidate_models):
        params = cand["params"]
        model = cand["model"]

        fit_start = time.perf_counter()
        model.fit(X_train, y_train)
        fit_time = time.perf_counter() - fit_start

        val_pred = model.predict(X_val)

        val_metrics = evaluate_predictions(y_val, val_pred)

        row = {
            "candidate_id": cand_id,
            "model_name": model_name,
            "params_json": json.dumps(params),
            "fit_time_sec": fit_time,
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_weighted_f1": val_metrics["weighted_f1"],
        }
        rows.append(row)

        if best is None:
            best = {
                "candidate_id": cand_id,
                "model": model,
                "params": params,
                "val_metrics": val_metrics,
                "fit_time_sec": fit_time,
            }
        else:
            old = best["val_metrics"]
            if (val_metrics["macro_f1"], val_metrics["accuracy"]) > (old["macro_f1"], old["accuracy"]):
                best = {
                    "candidate_id": cand_id,
                    "model": model,
                    "params": params,
                    "val_metrics": val_metrics,
                    "fit_time_sec": fit_time,
                }

    search_time = time.perf_counter() - search_start

    val_df = pd.DataFrame(rows)
    val_df.to_csv(
        RUN_RESULTS_DIR / f"{RUN_NAME}_{model_name_safe}_validation_grid.csv",
        index=False,
    )

    print("\nValidation grid:")
    print(val_df.sort_values(["val_macro_f1", "val_accuracy"], ascending=False).head(10).to_string(index=False))

    print("\nSelected:")
    print(json.dumps(best["params"], indent=2))
    print(f"Best val Macro-F1: {best['val_metrics']['macro_f1']:.4f}")
    print(f"Best val Accuracy: {best['val_metrics']['accuracy']:.4f}")

    # Refit selected model on train + val for final test.
    X_dev = np.vstack([X_train, X_val])
    y_dev = np.concatenate([y_train, y_val])

    final_model = best["model"]

    refit_start = time.perf_counter()
    final_model.fit(X_dev, y_dev)
    refit_time = time.perf_counter() - refit_start

    pred_start = time.perf_counter()
    y_pred = final_model.predict(X_test)
    predict_time = time.perf_counter() - pred_start

    test_metrics = evaluate_predictions(y_test, y_pred)

    save_model_outputs(model_name_safe, y_test, y_pred)

    result = {
        "model_name": model_name,
        "model_name_safe": model_name_safe,
        "selected_candidate_id": int(best["candidate_id"]),
        "selected_params_json": json.dumps(best["params"]),
        "val_accuracy": float(best["val_metrics"]["accuracy"]),
        "val_macro_f1": float(best["val_metrics"]["macro_f1"]),
        "val_weighted_f1": float(best["val_metrics"]["weighted_f1"]),
        "test_accuracy": float(test_metrics["accuracy"]),
        "test_macro_f1": float(test_metrics["macro_f1"]),
        "test_weighted_f1": float(test_metrics["weighted_f1"]),
        "search_time_sec": float(search_time),
        "refit_time_sec": float(refit_time),
        "predict_time_sec": float(predict_time),
        "num_train": int(len(y_train)),
        "num_val": int(len(y_val)),
        "num_dev_refit": int(len(y_dev)),
        "num_test": int(len(y_test)),
    }

    print("\nFinal test result:")
    for k, v in result.items():
        print(f"{k}: {v}")

    return result


# ============================================================
# 6. CANDIDATE MODEL BUILDERS
# ============================================================

def build_linear_svm_candidates():
    candidates = []

    for C in LINEAR_C_GRID:
        model = Pipeline([
            ("scaler", StandardScaler()),
            (
                "clf",
                LinearSVC(
                    C=C,
                    class_weight="balanced",
                    max_iter=20000,
                    random_state=SEED,
                    dual="auto",
                ),
            ),
        ])
        candidates.append({
            "params": {"C": C},
            "model": model,
        })

    return candidates


def build_rbf_svm_candidates():
    candidates = []

    for C in RBF_C_GRID:
        for gamma in RBF_GAMMA_GRID:
            model = Pipeline([
                ("scaler", StandardScaler()),
                (
                    "clf",
                    SVC(
                        kernel="rbf",
                        C=C,
                        gamma=gamma,
                        class_weight="balanced",
                    ),
                ),
            ])
            candidates.append({
                "params": {"C": C, "gamma": gamma},
                "model": model,
            })

    return candidates


def build_random_forest_candidates():
    candidates = []

    for n_estimators in RF_N_ESTIMATORS_GRID:
        for max_depth in RF_MAX_DEPTH_GRID:
            for min_samples_leaf in RF_MIN_SAMPLES_LEAF_GRID:
                model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    class_weight="balanced_subsample",
                    random_state=SEED,
                    n_jobs=-1,
                )

                candidates.append({
                    "params": {
                        "n_estimators": n_estimators,
                        "max_depth": max_depth,
                        "min_samples_leaf": min_samples_leaf,
                    },
                    "model": model,
                })

    return candidates


def build_xgboost_candidates():
    try:
        from xgboost import XGBClassifier
    except ImportError:
        print("XGBoost is not installed. Skipping XGBoost.")
        return []

    candidates = []

    for n_estimators in XGB_N_ESTIMATORS_GRID:
        for max_depth in XGB_MAX_DEPTH_GRID:
            for learning_rate in XGB_LEARNING_RATE_GRID:
                for subsample in XGB_SUBSAMPLE_GRID:

                    model = XGBClassifier(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        learning_rate=learning_rate,
                        subsample=subsample,
                        colsample_bytree=0.9,
                        objective="multi:softprob",
                        num_class=N_CLASSES,
                        eval_metric="mlogloss",
                        tree_method="hist",
                        random_state=SEED,
                        n_jobs=-1,
                    )

                    candidates.append({
                        "params": {
                            "n_estimators": n_estimators,
                            "max_depth": max_depth,
                            "learning_rate": learning_rate,
                            "subsample": subsample,
                            "colsample_bytree": 0.9,
                            "tree_method": "hist",
                        },
                        "model": model,
                    })

    return candidates


# ============================================================
# 7. MAIN
# ============================================================

def main():
    total_start = time.perf_counter()

    print("=" * 80)
    print("Matched classical baselines: 12C-50/class split")
    print("=" * 80)

    save_system_info()

    X_train, y_train, X_val, y_val, X_test, y_test = load_saved_split()

    all_results = []

    experiments = [
        ("Linear SVM", "linear_svm", build_linear_svm_candidates()),
        ("RBF-SVM", "rbf_svm", build_rbf_svm_candidates()),
        ("Random Forest", "random_forest", build_random_forest_candidates()),
        ("XGBoost", "xgboost", build_xgboost_candidates()),
    ]

    for model_name, model_name_safe, candidates in experiments:
        if len(candidates) == 0:
            continue

        result = fit_select_test_model(
            model_name=model_name,
            model_name_safe=model_name_safe,
            candidate_models=candidates,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
        )

        all_results.append(result)

        pd.DataFrame(all_results).to_csv(
            RUN_RESULTS_DIR / f"{RUN_NAME}_all_results_partial.csv",
            index=False,
        )

    all_df = pd.DataFrame(all_results)
    all_df = all_df.sort_values(
        ["test_macro_f1", "test_accuracy"],
        ascending=False,
    )

    all_df.to_csv(
        RUN_RESULTS_DIR / f"{RUN_NAME}_all_results.csv",
        index=False,
    )

    total_time = time.perf_counter() - total_start

    summary = {
        "run_name": RUN_NAME,
        "total_wall_time_sec": float(total_time),
        "best_model_by_test_macro_f1": all_df.iloc[0]["model_name"] if len(all_df) else None,
        "best_test_macro_f1": float(all_df.iloc[0]["test_macro_f1"]) if len(all_df) else None,
        "best_test_accuracy": float(all_df.iloc[0]["test_accuracy"]) if len(all_df) else None,
    }

    with open(RUN_RESULTS_DIR / f"{RUN_NAME}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print("ALL MATCHED CLASSICAL RESULTS")
    print("=" * 80)
    print(all_df.to_string(index=False))

    print("\nSummary:")
    print(json.dumps(summary, indent=2))

    print(f"\nSaved results to: {RUN_RESULTS_DIR}")


if __name__ == "__main__":
    main()
