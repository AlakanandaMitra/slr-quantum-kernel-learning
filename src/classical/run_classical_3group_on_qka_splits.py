from __future__ import annotations

import json
import time
import platform
import subprocess
import warnings
from pathlib import Path
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.colors import LinearSegmentedColormap

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False

import sys
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


from src.paths import RESULTS_DIR
# ============================================================
# 1. CONFIG
# ============================================================

SEED = 42

BASE_RESULTS_DIR = RESULTS_DIR / "qka_corrected"

OUT_DIR = BASE_RESULTS_DIR / "classical_3group_on_qka_splits"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GROUP_NAMES = {
    0: "Healthy",
    1: "Single",
    2: "Multiple",
}

# These should match your QKA run folders and run names.
# Adjust folder names only if yours are slightly different.
QKA_SPLIT_RUNS = {
    "C-3G-100": {
        "qka_dir": BASE_RESULTS_DIR / "qka_adaptive_soft_3group_corrected_100",
        "run_name": "qka_adaptive_soft_3group_corrected_100",
        "train_per_group": 100,
    },
    "C-3G-150": {
        "qka_dir": BASE_RESULTS_DIR / "qka_adaptive_soft_3group_corrected_150",
        "run_name": "qka_adaptive_soft_3group_corrected_150",
        "train_per_group": 150,
    },
    "C-3G-210": {
        "qka_dir": BASE_RESULTS_DIR / "qka_adaptive_soft_3group_corrected_210",
        "run_name": "qka_adaptive_soft_3group_corrected_210",
        "train_per_group": 210,
    },
}

# Hyperparameter grids.
LINEAR_SVM_C_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]
RBF_SVM_C_GRID = [0.1, 1.0, 10.0, 100.0]
RBF_SVM_GAMMA_GRID = ["scale", 0.01, 0.1, 1.0]

RF_N_ESTIMATORS_GRID = [200, 500]
RF_MAX_DEPTH_GRID = [None, 10, 20]
RF_MIN_SAMPLES_LEAF_GRID = [1, 2, 5]

XGB_N_ESTIMATORS_GRID = [200, 500]
XGB_MAX_DEPTH_GRID = [3, 5]
XGB_LEARNING_RATE_GRID = [0.03, 0.1]
XGB_SUBSAMPLE_GRID = [0.8, 1.0]

USE_XGBOOST_GPU = True
XGB_N_JOBS = 1


# ============================================================
# 2. PLOTTING
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


def save_confusion_matrix(cm: np.ndarray, out_base: Path) -> None:
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
# 3. SYSTEM INFO
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
        "base_results_dir": str(BASE_RESULTS_DIR),
        "out_dir": str(OUT_DIR),
        "has_xgboost": HAS_XGBOOST,
        "use_xgboost_gpu": USE_XGBOOST_GPU,
        "qka_split_runs": {
            k: {
                "qka_dir": str(v["qka_dir"]),
                "run_name": v["run_name"],
                "train_per_group": v["train_per_group"],
            }
            for k, v in QKA_SPLIT_RUNS.items()
        },
    }

    with open(OUT_DIR / "classical_3group_on_qka_splits_system_info.json", "w") as f:
        json.dump(info, f, indent=2)

    with open(OUT_DIR / "classical_3group_on_qka_splits_nvidia_smi.txt", "w") as f:
        f.write(get_nvidia_smi_text())


# ============================================================
# 4. DATA LOADING
# ============================================================

def load_qka_split(qka_dir: Path, run_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    files = {
        "X_train": qka_dir / f"{run_name}_X_train_pca64_lda11.npy",
        "y_train": qka_dir / f"{run_name}_y_train_3group.npy",
        "X_val": qka_dir / f"{run_name}_X_val_pca64_lda11.npy",
        "y_val": qka_dir / f"{run_name}_y_val_3group.npy",
        "X_test": qka_dir / f"{run_name}_X_test_pca64_lda11.npy",
        "y_test": qka_dir / f"{run_name}_y_test_3group.npy",
    }

    missing = [str(p) for p in files.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing saved QKA split files:\n"
            + "\n".join(missing)
            + "\n\nCheck the qka_dir and run_name in QKA_SPLIT_RUNS."
        )

    X_train = np.load(files["X_train"])
    y_train = np.load(files["y_train"])
    X_val = np.load(files["X_val"])
    y_val = np.load(files["y_val"])
    X_test = np.load(files["X_test"])
    y_test = np.load(files["y_test"])

    return X_train, y_train, X_val, y_val, X_test, y_test


# ============================================================
# 5. MODEL BUILDERS
# ============================================================

def make_linear_svm(C: float):
    return LinearSVC(
        C=C,
        class_weight="balanced",
        random_state=SEED,
        max_iter=20000,
        dual="auto",
    )


def make_rbf_svm(C: float, gamma):
    return SVC(
        kernel="rbf",
        C=C,
        gamma=gamma,
        class_weight="balanced",
        random_state=SEED,
    )


def make_random_forest(n_estimators: int, max_depth, min_samples_leaf: int):
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
    )


def make_xgboost(n_estimators: int, max_depth: int, learning_rate: float, subsample: float):
    if not HAS_XGBOOST:
        raise RuntimeError("xgboost is not installed in this environment.")

    params = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "subsample": subsample,
        "colsample_bytree": 1.0,
        "objective": "multi:softprob",
        "num_class": 3,
        "eval_metric": "mlogloss",
        "random_state": SEED,
        "n_jobs": XGB_N_JOBS,
    }

    if USE_XGBOOST_GPU:
        # XGBoost 2.x style. If your environment is older, change to:
        # params.update({"tree_method": "gpu_hist"})
        params.update({
            "tree_method": "hist",
            "device": "cuda",
        })
    else:
        params.update({
            "tree_method": "hist",
        })

    return XGBClassifier(**params)


# ============================================================
# 6. VALIDATION SELECTION
# ============================================================

def evaluate_predictions(y_true, y_pred) -> Dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def select_best_model(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> Tuple[Any, Dict[str, Any], pd.DataFrame]:

    rows: List[Dict[str, Any]] = []
    best_model = None
    best_params: Dict[str, Any] = {}
    best_score = (-1.0, -1.0)

    if model_name == "Linear SVM":
        for C in LINEAR_SVM_C_GRID:
            params = {"C": C}
            model = make_linear_svm(C)
            start = time.perf_counter()
            model.fit(X_train, y_train)
            fit_time = time.perf_counter() - start

            y_val_pred = model.predict(X_val)
            metrics = evaluate_predictions(y_val, y_val_pred)

            row = {
                "model": model_name,
                **params,
                **{f"val_{k}": v for k, v in metrics.items()},
                "fit_time_sec": fit_time,
            }
            rows.append(row)

            score = (metrics["macro_f1"], metrics["accuracy"])
            if score > best_score:
                best_score = score
                best_model = model
                best_params = params

    elif model_name == "RBF-SVM":
        for C in RBF_SVM_C_GRID:
            for gamma in RBF_SVM_GAMMA_GRID:
                params = {"C": C, "gamma": gamma}
                model = make_rbf_svm(C, gamma)
                start = time.perf_counter()
                model.fit(X_train, y_train)
                fit_time = time.perf_counter() - start

                y_val_pred = model.predict(X_val)
                metrics = evaluate_predictions(y_val, y_val_pred)

                row = {
                    "model": model_name,
                    **params,
                    **{f"val_{k}": v for k, v in metrics.items()},
                    "fit_time_sec": fit_time,
                }
                rows.append(row)

                score = (metrics["macro_f1"], metrics["accuracy"])
                if score > best_score:
                    best_score = score
                    best_model = model
                    best_params = params

    elif model_name == "Random Forest":
        for n_estimators in RF_N_ESTIMATORS_GRID:
            for max_depth in RF_MAX_DEPTH_GRID:
                for min_samples_leaf in RF_MIN_SAMPLES_LEAF_GRID:
                    params = {
                        "n_estimators": n_estimators,
                        "max_depth": max_depth,
                        "min_samples_leaf": min_samples_leaf,
                    }
                    model = make_random_forest(n_estimators, max_depth, min_samples_leaf)
                    start = time.perf_counter()
                    model.fit(X_train, y_train)
                    fit_time = time.perf_counter() - start

                    y_val_pred = model.predict(X_val)
                    metrics = evaluate_predictions(y_val, y_val_pred)

                    row = {
                        "model": model_name,
                        **params,
                        **{f"val_{k}": v for k, v in metrics.items()},
                        "fit_time_sec": fit_time,
                    }
                    rows.append(row)

                    score = (metrics["macro_f1"], metrics["accuracy"])
                    if score > best_score:
                        best_score = score
                        best_model = model
                        best_params = params

    elif model_name == "XGBoost":
        if not HAS_XGBOOST:
            raise RuntimeError("XGBoost requested but not installed.")

        for n_estimators in XGB_N_ESTIMATORS_GRID:
            for max_depth in XGB_MAX_DEPTH_GRID:
                for learning_rate in XGB_LEARNING_RATE_GRID:
                    for subsample in XGB_SUBSAMPLE_GRID:
                        params = {
                            "n_estimators": n_estimators,
                            "max_depth": max_depth,
                            "learning_rate": learning_rate,
                            "subsample": subsample,
                        }
                        model = make_xgboost(
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            learning_rate=learning_rate,
                            subsample=subsample,
                        )

                        start = time.perf_counter()
                        model.fit(X_train, y_train)
                        fit_time = time.perf_counter() - start

                        y_val_pred = model.predict(X_val)
                        metrics = evaluate_predictions(y_val, y_val_pred)

                        row = {
                            "model": model_name,
                            **params,
                            **{f"val_{k}": v for k, v in metrics.items()},
                            "fit_time_sec": fit_time,
                        }
                        rows.append(row)

                        score = (metrics["macro_f1"], metrics["accuracy"])
                        if score > best_score:
                            best_score = score
                            best_model = model
                            best_params = params

    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    return best_model, best_params, pd.DataFrame(rows)


# ============================================================
# 7. RUN ONE SPLIT
# ============================================================

def run_one_split(exp_id: str, cfg: Dict[str, Any]) -> pd.DataFrame:
    qka_dir = cfg["qka_dir"]
    run_name = cfg["run_name"]
    train_per_group = cfg["train_per_group"]

    split_out_dir = OUT_DIR / exp_id
    split_out_dir.mkdir(parents=True, exist_ok=True)

    X_train, y_train, X_val, y_val, X_test, y_test = load_qka_split(qka_dir, run_name)

    print("\n" + "=" * 80)
    print(f"Running {exp_id}")
    print("=" * 80)
    print(f"Source QKA dir : {qka_dir}")
    print(f"Source run name: {run_name}")
    print(f"Train shape: {X_train.shape}, counts={np.bincount(y_train, minlength=3)}")
    print(f"Val shape  : {X_val.shape}, counts={np.bincount(y_val, minlength=3)}")
    print(f"Test shape : {X_test.shape}, counts={np.bincount(y_test, minlength=3)}")

    split_info = {
        "exp_id": exp_id,
        "source_qka_dir": str(qka_dir),
        "source_run_name": run_name,
        "train_per_group": train_per_group,
        "num_train": int(len(y_train)),
        "num_val": int(len(y_val)),
        "num_test": int(len(y_test)),
        "train_counts": np.bincount(y_train, minlength=3).tolist(),
        "val_counts": np.bincount(y_val, minlength=3).tolist(),
        "test_counts": np.bincount(y_test, minlength=3).tolist(),
    }

    with open(split_out_dir / f"{exp_id}_split_info.json", "w") as f:
        json.dump(split_info, f, indent=2)

    model_names = ["Linear SVM", "RBF-SVM", "Random Forest"]
    if HAS_XGBOOST:
        model_names.append("XGBoost")

    all_summary_rows = []

    for model_name in model_names:
        print(f"\nSelecting {model_name} on validation set...")

        model_start = time.perf_counter()

        best_model, best_params, val_df = select_best_model(
            model_name,
            X_train,
            y_train,
            X_val,
            y_val,
        )

        val_df.to_csv(
            split_out_dir / f"{exp_id}_{model_name.replace(' ', '_').replace('-', '_')}_validation_grid.csv",
            index=False,
        )

        y_test_pred = best_model.predict(X_test)
        test_metrics = evaluate_predictions(y_test, y_test_pred)

        cm = confusion_matrix(y_test, y_test_pred, labels=[0, 1, 2])

        safe_model_name = model_name.replace(" ", "_").replace("-", "_")
        save_confusion_matrix(
            cm,
            split_out_dir / f"{exp_id}_{safe_model_name}_confusion_matrix_counts_pretty_largefont_notitle",
        )

        report_dict = classification_report(
            y_test,
            y_test_pred,
            target_names=[GROUP_NAMES[i] for i in range(3)],
            output_dict=True,
            zero_division=0,
        )

        report_text = classification_report(
            y_test,
            y_test_pred,
            target_names=[GROUP_NAMES[i] for i in range(3)],
            zero_division=0,
        )

        pd.DataFrame(report_dict).transpose().to_csv(
            split_out_dir / f"{exp_id}_{safe_model_name}_classification_report.csv"
        )

        with open(split_out_dir / f"{exp_id}_{safe_model_name}_classification_report.txt", "w") as f:
            f.write(report_text)

        np.save(
            split_out_dir / f"{exp_id}_{safe_model_name}_predictions.npy",
            y_test_pred,
        )

        model_time = time.perf_counter() - model_start

        row = {
            "experiment_id": exp_id,
            "train_per_group": train_per_group,
            "model": model_name,
            "num_train": int(len(y_train)),
            "num_val": int(len(y_val)),
            "num_test": int(len(y_test)),
            "test_accuracy": test_metrics["accuracy"],
            "test_macro_f1": test_metrics["macro_f1"],
            "test_weighted_f1": test_metrics["weighted_f1"],
            "selected_params": json.dumps(best_params),
            "model_total_time_sec": model_time,
        }

        all_summary_rows.append(row)

        print(f"{model_name} selected params: {best_params}")
        print(
            f"{model_name} test: "
            f"acc={test_metrics['accuracy']:.4f}, "
            f"macro_f1={test_metrics['macro_f1']:.4f}, "
            f"weighted_f1={test_metrics['weighted_f1']:.4f}"
        )

    summary_df = pd.DataFrame(all_summary_rows)
    summary_df.to_csv(split_out_dir / f"{exp_id}_classical_summary.csv", index=False)

    return summary_df


# ============================================================
# 8. MAIN
# ============================================================

def main():
    total_start = time.perf_counter()
    save_system_info()

    all_results = []

    for exp_id, cfg in QKA_SPLIT_RUNS.items():
        try:
            df = run_one_split(exp_id, cfg)
            all_results.append(df)
        except FileNotFoundError as e:
            print("\n" + "!" * 80)
            print(f"Skipping {exp_id} because required files were not found.")
            print(str(e))
            print("!" * 80)

    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        final_df.to_csv(OUT_DIR / "classical_3group_on_qka_splits_all_results.csv", index=False)

        pivot = final_df.pivot_table(
            index=["experiment_id", "train_per_group", "model"],
            values=["test_accuracy", "test_macro_f1", "test_weighted_f1"],
            aggfunc="first",
        ).reset_index()

        pivot.to_csv(OUT_DIR / "classical_3group_on_qka_splits_pivot.csv", index=False)

        print("\n" + "=" * 80)
        print("ALL CLASSICAL 3-GROUP MATCHED RESULTS")
        print("=" * 80)
        print(pivot.to_string(index=False))

    print(f"\nTotal wall time: {time.perf_counter() - total_start:.2f} sec")
    print(f"Saved all results to: {OUT_DIR}")


if __name__ == "__main__":
    main()
