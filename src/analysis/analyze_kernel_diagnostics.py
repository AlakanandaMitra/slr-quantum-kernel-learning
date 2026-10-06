#!/usr/bin/env python3
"""Five-seed kernel diagnostics for fixed QK and gamma-only QKA.

The script reuses the saved raw training kernels and matched labels at one
training size (default: 150 samples/group). It computes:

* hard- and soft-target kernel alignment;
* within-group and between-group mean similarity;
* pair-specific similarities (Healthy-Single, Healthy-Multiple,
  Single-Multiple);
* eigenvalue spectrum;
* entropy effective rank and stable rank;
* positive-spectrum condition number.

No quantum kernels are recomputed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.paths import RESULTS_DIR

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_GAMMA_ROOT = RESULTS_DIR / "qka_gamma_only"
DEFAULT_BASELINE_ROOT = RESULTS_DIR / "multiseed_matched"
DEFAULT_OUTPUT_ROOT = RESULTS_DIR / "kernel_diagnostics"


SEEDS = (42, 52, 62, 72, 82)
GROUP_NAMES = {0: "Healthy", 1: "Single", 2: "Multiple"}
MODELS = ("Fixed QK", "Gamma-only QKA")

MODEL_COLORS = {
    "Fixed QK": "#7FA9AA",
    "Gamma-only QKA": "#E7B7C8",
}
MODEL_MARKERS = {
    "Fixed QK": "s",
    "Gamma-only QKA": "o",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-per-group", type=int, default=150)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--qka-root", type=Path, default=DEFAULT_QKA_ROOT)
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--eigenvalue-tolerance",
        type=float,
        default=1e-10,
        help="Relative tolerance used to identify positive eigenvalues.",
    )
    return parser.parse_args()


def qka_paths(root: Path, size: int, seed: int) -> tuple[Path, Path, Path]:
    name = f"qka_gamma_only_soft_3group_{size}_seed{seed}"
    directory = root / name
    return (
        directory / f"{name}_kernel_train_raw_final.npy",
        directory / f"{name}_y_train_3group.npy",
        directory / f"{name}_results_summary.json",
    )


def fixed_paths(root: Path, size: int, seed: int) -> tuple[Path, Path]:
    name = f"fixed_qk_3group_{size}_seed{seed}"
    directory = root / f"train{size}_seed{seed}"
    return (
        directory / f"{name}_kernel_train_raw.npy",
        directory / f"{name}_results_summary.json",
    )


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return path


def soft_target(y: np.ndarray) -> np.ndarray:
    values = np.array(
        [
            [1.0, 0.2, 0.0],
            [0.2, 1.0, 0.5],
            [0.0, 0.5, 1.0],
        ],
        dtype=np.float64,
    )
    return values[y[:, None], y[None, :]]


def hard_target(y: np.ndarray) -> np.ndarray:
    return (y[:, None] == y[None, :]).astype(np.float64)


def centered_alignment(K: np.ndarray, T: np.ndarray) -> float:
    """Centered Frobenius kernel-target alignment."""
    n = K.shape[0]
    H = np.eye(n, dtype=np.float64) - np.ones((n, n), dtype=np.float64) / n
    Kc = H @ K @ H
    Tc = H @ T @ H
    denominator = np.linalg.norm(Kc, "fro") * np.linalg.norm(Tc, "fro")
    return float(np.sum(Kc * Tc) / denominator) if denominator > 0 else float("nan")


def uncentered_alignment(K: np.ndarray, T: np.ndarray) -> float:
    denominator = np.linalg.norm(K, "fro") * np.linalg.norm(T, "fro")
    return float(np.sum(K * T) / denominator) if denominator > 0 else float("nan")


def sanitize_kernel(K: np.ndarray) -> np.ndarray:
    K = np.asarray(K, dtype=np.float64)
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError(f"Expected a square kernel matrix, got shape {K.shape}")
    if not np.all(np.isfinite(K)):
        raise ValueError("Kernel contains NaN or infinite values")
    return 0.5 * (K + K.T)


def masked_mean(K: np.ndarray, mask: np.ndarray) -> float:
    values = K[mask]
    return float(values.mean()) if values.size else float("nan")


def similarity_statistics(K: np.ndarray, y: np.ndarray) -> dict[str, float]:
    n = len(y)
    off_diagonal = ~np.eye(n, dtype=bool)
    same = y[:, None] == y[None, :]

    result: dict[str, float] = {
        "within_group_similarity": masked_mean(K, same & off_diagonal),
        "between_group_similarity": masked_mean(K, (~same) & off_diagonal),
    }
    result["within_minus_between"] = (
        result["within_group_similarity"] - result["between_group_similarity"]
    )

    for group_id, group_name in GROUP_NAMES.items():
        mask = (y[:, None] == group_id) & (y[None, :] == group_id) & off_diagonal
        result[f"within_{group_name.lower()}_similarity"] = masked_mean(K, mask)

    pairs = ((0, 1), (0, 2), (1, 2))
    for a, b in pairs:
        mask = ((y[:, None] == a) & (y[None, :] == b)) | (
            (y[:, None] == b) & (y[None, :] == a)
        )
        name = f"between_{GROUP_NAMES[a].lower()}_{GROUP_NAMES[b].lower()}_similarity"
        result[name] = masked_mean(K, mask)

    return result


def spectral_statistics(
    K: np.ndarray, relative_tolerance: float
) -> tuple[dict[str, float], np.ndarray]:
    eigenvalues = np.linalg.eigvalsh(K)
    eigenvalues = np.sort(eigenvalues)[::-1]

    largest_abs = max(float(np.max(np.abs(eigenvalues))), 1.0)
    tolerance = relative_tolerance * largest_abs
    positive = eigenvalues[eigenvalues > tolerance]
    clipped = np.clip(eigenvalues, 0.0, None)

    trace = float(clipped.sum())
    if trace > 0:
        probabilities = clipped / trace
        nz = probabilities[probabilities > 0]
        effective_rank = float(np.exp(-np.sum(nz * np.log(nz))))
    else:
        effective_rank = float("nan")

    fro_sq = float(np.sum(clipped**2))
    stable_rank = float(trace**2 / fro_sq) if fro_sq > 0 else float("nan")
    condition_number = (
        float(positive.max() / positive.min()) if positive.size >= 2 else float("nan")
    )

    negative = eigenvalues[eigenvalues < -tolerance]
    stats = {
        "trace": trace,
        "largest_eigenvalue": float(eigenvalues[0]),
        "smallest_eigenvalue": float(eigenvalues[-1]),
        "positive_eigenvalue_count": int(positive.size),
        "negative_eigenvalue_count": int(negative.size),
        "effective_rank": effective_rank,
        "stable_rank": stable_rank,
        "condition_number_positive": condition_number,
        "top1_trace_fraction": float(clipped[0] / trace) if trace > 0 else float("nan"),
        "top5_trace_fraction": float(clipped[:5].sum() / trace) if trace > 0 else float("nan"),
        "top10_trace_fraction": float(clipped[:10].sum() / trace) if trace > 0 else float("nan"),
    }
    return stats, eigenvalues


def load_test_macro_f1(summary_path: Path) -> float:
    if not summary_path.exists():
        return float("nan")
    with open(summary_path, "r", encoding="utf-8") as handle:
        summary = json.load(handle)
    return float(summary.get("test_macro_f1", float("nan")))


def analyze_one(
    model: str,
    seed: int,
    size: int,
    K: np.ndarray,
    y: np.ndarray,
    test_macro_f1: float,
    eigenvalue_tolerance: float,
) -> tuple[dict[str, float | int | str], np.ndarray]:
    K = sanitize_kernel(K)
    if K.shape[0] != len(y):
        raise ValueError(
            f"Kernel/label mismatch for {model}, seed {seed}: {K.shape[0]} vs {len(y)}"
        )

    hard = hard_target(y)
    soft = soft_target(y)

    row: dict[str, float | int | str] = {
        "model": model,
        "train_per_group": size,
        "seed": seed,
        "num_train": len(y),
        "test_macro_f1": test_macro_f1,
        "hard_alignment_uncentered": uncentered_alignment(K, hard),
        "soft_alignment_uncentered": uncentered_alignment(K, soft),
        "hard_alignment_centered": centered_alignment(K, hard),
        "soft_alignment_centered": centered_alignment(K, soft),
        "mean_diagonal": float(np.mean(np.diag(K))),
    }
    row.update(similarity_statistics(K, y))
    spectral, eigenvalues = spectral_statistics(K, eigenvalue_tolerance)
    row.update(spectral)
    return row, eigenvalues


def aggregate_mean_std(df: pd.DataFrame) -> pd.DataFrame:
    numeric = [
        column
        for column in df.select_dtypes(include=[np.number]).columns
        if column not in {"seed", "train_per_group", "num_train"}
    ]
    rows: list[dict[str, float | int | str]] = []
    for model, group in df.groupby("model", sort=False):
        row: dict[str, float | int | str] = {
            "model": model,
            "train_per_group": int(group["train_per_group"].iloc[0]),
            "n_seeds": int(group["seed"].nunique()),
        }
        for column in numeric:
            row[f"{column}_mean"] = float(group[column].mean())
            row[f"{column}_std"] = float(group[column].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def paired_differences(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "soft_alignment_centered",
        "hard_alignment_centered",
        "within_group_similarity",
        "between_group_similarity",
        "within_minus_between",
        "effective_rank",
        "stable_rank",
        "condition_number_positive",
        "top5_trace_fraction",
    ]
    rows = []
    for metric in metrics:
        pivot = df.pivot(index="seed", columns="model", values=metric).dropna()
        if not set(MODELS).issubset(pivot.columns):
            continue
        difference = pivot["Gamma-only QKA"] - pivot["Fixed QK"]
        rows.append(
            {
                "metric": metric,
                "n_paired_seeds": len(difference),
                "mean_difference_gamma_minus_fixed": float(difference.mean()),
                "std_difference": float(difference.std(ddof=1)),
                "min_difference": float(difference.min()),
                "max_difference": float(difference.max()),
            }
        )
    return pd.DataFrame(rows)


def save_alignment_plot(df: pd.DataFrame, output_dir: Path) -> None:
    metrics = ["hard_alignment_centered", "soft_alignment_centered"]
    labels = ["Hard target", "Soft target"]
    x = np.arange(len(metrics), dtype=float)
    offsets = {"Fixed QK": -0.12, "Gamma-only QKA": 0.12}

    fig, ax = plt.subplots(figsize=(6.5, 4.7))
    for model in MODELS:
        model_df = df[df["model"] == model]
        means = np.array([model_df[m].mean() for m in metrics])
        sds = np.array([model_df[m].std(ddof=1) for m in metrics])
        ax.errorbar(
            x + offsets[model],
            means,
            yerr=sds,
            marker=MODEL_MARKERS[model],
            color=MODEL_COLORS[model],
            markeredgecolor="black",
            markeredgewidth=0.5,
            linewidth=1.7,
            capsize=5,
            label=model,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Centered kernel-target alignment") #, fontweight="bold")
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "kernel_alignment_comparison.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "kernel_alignment_comparison.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def save_similarity_plot(df: pd.DataFrame, output_dir: Path) -> None:
    metrics = ["within_group_similarity", "between_group_similarity"]
    labels = ["Within group", "Between groups"]
    x = np.arange(len(metrics), dtype=float)
    width = 0.34

    fig, ax = plt.subplots(figsize=(6.5, 4.7))
    for index, model in enumerate(MODELS):
        model_df = df[df["model"] == model]
        means = np.array([model_df[m].mean() for m in metrics])
        sds = np.array([model_df[m].std(ddof=1) for m in metrics])
        positions = x + (index - 0.5) * width
        ax.bar(
            positions,
            means,
            yerr=sds,
            width=width,
            capsize=5,
            color=MODEL_COLORS[model],
            edgecolor="black",
            linewidth=0.6,
            label=model,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean kernel similarity") #, fontweight="bold")
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "within_between_similarity.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "within_between_similarity.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def save_eigenspectrum_plot(eigen_df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.7, 4.8))
    for model in MODELS:
        model_df = eigen_df[eigen_df["model"] == model]
        pivot = model_df.pivot(index="eigen_index", columns="seed", values="normalized_eigenvalue")
        means = pivot.mean(axis=1)
        sds = pivot.std(axis=1, ddof=1)
        x = means.index.to_numpy() + 1
        keep = x <= min(50, len(x))
        ax.plot(
            x[keep],
            means.to_numpy()[keep],
            marker=MODEL_MARKERS[model],
            markersize=3.5,
            linewidth=1.5,
            color=MODEL_COLORS[model],
            label=model,
        )
        ax.fill_between(
            x[keep],
            np.maximum(means.to_numpy()[keep] - sds.to_numpy()[keep], 1e-12),
            means.to_numpy()[keep] + sds.to_numpy()[keep],
            color=MODEL_COLORS[model],
            alpha=0.18,
        )
    ax.set_yscale("log")
    ax.set_xlabel("Eigenvalue index") #, fontweight="bold")
    ax.set_ylabel("Normalized eigenvalue") #, fontweight="bold")
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "kernel_eigenvalue_spectrum.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "kernel_eigenvalue_spectrum.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    diagnostic_rows: list[dict[str, float | int | str]] = []
    eigen_rows: list[dict[str, float | int | str]] = []

    for seed in args.seeds:
        qka_kernel_path, label_path, qka_summary_path = qka_paths(
            args.qka_root, args.train_per_group, seed
        )
        fixed_kernel_path, fixed_summary_path = fixed_paths(
            args.baseline_root, args.train_per_group, seed
        )

        y = np.load(require_file(label_path)).astype(int)
        inputs = [
            (
                "Fixed QK",
                np.load(require_file(fixed_kernel_path)),
                load_test_macro_f1(fixed_summary_path),
            ),
            (
                "Gamma-only QKA",
                np.load(require_file(qka_kernel_path)),
                load_test_macro_f1(qka_summary_path),
            ),
        ]

        for model, K, test_macro_f1 in inputs:
            row, eigenvalues = analyze_one(
                model=model,
                seed=seed,
                size=args.train_per_group,
                K=K,
                y=y,
                test_macro_f1=test_macro_f1,
                eigenvalue_tolerance=args.eigenvalue_tolerance,
            )
            diagnostic_rows.append(row)

            clipped = np.clip(eigenvalues, 0.0, None)
            denominator = clipped.sum()
            normalized = clipped / denominator if denominator > 0 else clipped
            for index, (eigenvalue, normalized_value) in enumerate(
                zip(eigenvalues, normalized)
            ):
                eigen_rows.append(
                    {
                        "model": model,
                        "train_per_group": args.train_per_group,
                        "seed": seed,
                        "eigen_index": index,
                        "eigenvalue": float(eigenvalue),
                        "normalized_eigenvalue": float(normalized_value),
                    }
                )

    diagnostics = pd.DataFrame(diagnostic_rows)
    eigen_df = pd.DataFrame(eigen_rows)
    summary = aggregate_mean_std(diagnostics)
    differences = paired_differences(diagnostics)

    diagnostics.to_csv(args.output_dir / "kernel_diagnostics_seed_results.csv", index=False)
    summary.to_csv(args.output_dir / "kernel_diagnostics_mean_std.csv", index=False)
    differences.to_csv(args.output_dir / "kernel_diagnostics_paired_differences.csv", index=False)
    eigen_df.to_csv(args.output_dir / "kernel_eigenvalue_spectra.csv", index=False)

    save_alignment_plot(diagnostics, args.output_dir)
    save_similarity_plot(diagnostics, args.output_dir)
    save_eigenspectrum_plot(eigen_df, args.output_dir)

    print("Created:")
    for filename in (
        "kernel_diagnostics_seed_results.csv",
        "kernel_diagnostics_mean_std.csv",
        "kernel_diagnostics_paired_differences.csv",
        "kernel_eigenvalue_spectra.csv",
        "kernel_alignment_comparison.pdf",
        "within_between_similarity.pdf",
        "kernel_eigenvalue_spectrum.pdf",
    ):
        print(args.output_dir / filename)


if __name__ == "__main__":
    main()
