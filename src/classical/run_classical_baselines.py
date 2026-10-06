from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.svm import SVC
from xgboost import XGBClassifier


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run classical baselines on PCA-reduced embeddings."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing train/valid/test PCA .npy files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/baselines",
        help="Directory to save metrics and predictions.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="run",
        help="Tag added to output filenames, e.g. efficientnet_b0_pca_16.",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed.",
    )
    return parser.parse_args()


def load_split(input_dir: Path, split: str):
    x_path = input_dir / f"{split}_embeddings_pca.npy"
    y_path = input_dir / f"{split}_label_idxs.npy"

    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(f"Missing files for split '{split}' in {input_dir}")

    X = np.load(x_path)
    y = np.load(y_path)

    print(f"Loaded {split}: X={X.shape}, y={y.shape}")
    return X, y


def maybe_load_split(input_dir: Path, split: str):
    x_path = input_dir / f"{split}_embeddings_pca.npy"
    y_path = input_dir / f"{split}_label_idxs.npy"

    if x_path.exists() and y_path.exists():
        return load_split(input_dir, split)
    return None


def evaluate_model(
    model_name: str,
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_eval)

    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_eval, y_pred),
        "macro_f1": f1_score(y_eval, y_pred, average="macro"),
        "weighted_f1": f1_score(y_eval, y_pred, average="weighted"),
    }

    report_dict = classification_report(y_eval, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_eval, y_pred)

    return metrics, y_pred, report_dict, cm


def save_confusion_matrix(cm: np.ndarray, out_path: Path):
    cm_df = pd.DataFrame(cm)
    cm_df.to_csv(out_path, index=False)


def save_predictions(y_true: np.ndarray, y_pred: np.ndarray, out_path: Path):
    pred_df = pd.DataFrame(
        {
            "y_true": y_true,
            "y_pred": y_pred,
        }
    )
    pred_df.to_csv(out_path, index=False)


def main():
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) / args.tag
    output_dir.mkdir(parents=True, exist_ok=True)

    X_train, y_train = load_split(input_dir, "train")

    eval_split = None
    valid_data = maybe_load_split(input_dir, "valid")
    if valid_data is None:
        valid_data = maybe_load_split(input_dir, "validation")

    test_data = maybe_load_split(input_dir, "test")

    if valid_data is not None:
        eval_split = "valid" if (input_dir / "valid_embeddings_pca.npy").exists() else "validation"
        X_eval, y_eval = valid_data
    elif test_data is not None:
        eval_split = "test"
        X_eval, y_eval = test_data
    else:
        raise FileNotFoundError("No valid/validation or test split found for evaluation.")

    print(f"\nUsing '{eval_split}' split for evaluation during this run.\n")

    n_classes = len(np.unique(y_train))

    models = {
        "linear_svm": SVC(
            kernel="linear",
            C=1.0,
            random_state=args.random_state,
        ),
        "rbf_svm": SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            probability=False,
            random_state=args.random_state,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            random_state=args.random_state,
            n_jobs=-1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softmax",
            num_class=n_classes,
            eval_metric="mlogloss",
            random_state=args.random_state,
            n_jobs=-1,
        ),
    }

    summary_rows = []

    for model_name, model in models.items():
        print(f"Running {model_name} ...")

        metrics, y_pred, report_dict, cm = evaluate_model(
            model_name=model_name,
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_eval=X_eval,
            y_eval=y_eval,
        )

        summary_rows.append(metrics)

        print(
            f"{model_name}: "
            f"accuracy={metrics['accuracy']:.4f}, "
            f"macro_f1={metrics['macro_f1']:.4f}, "
            f"weighted_f1={metrics['weighted_f1']:.4f}"
        )

        report_df = pd.DataFrame(report_dict).transpose()
        report_df.to_csv(output_dir / f"{model_name}_{eval_split}_classification_report.csv")

        save_confusion_matrix(
            cm=cm,
            out_path=output_dir / f"{model_name}_{eval_split}_confusion_matrix.csv",
        )

        save_predictions(
            y_true=y_eval,
            y_pred=y_pred,
            out_path=output_dir / f"{model_name}_{eval_split}_predictions.csv",
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(
        by=["macro_f1", "accuracy"], ascending=False
    )
    summary_path = output_dir / f"summary_{eval_split}.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\nSummary:")
    print(summary_df.to_string(index=False))
    print(f"\nSaved summary to: {summary_path}")

    if test_data is not None and eval_split != "test":
        print(
            "\nNote: this run evaluated on validation data. "
            "Use the best configuration later for final test evaluation."
        )


if __name__ == "__main__":
    main()