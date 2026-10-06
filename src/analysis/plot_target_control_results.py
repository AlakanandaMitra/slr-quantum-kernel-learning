#!/usr/bin/env python3
"""Create target-control and alignment-versus-performance figures."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.paths import RESULTS_DIR

AGGREGATE_DIR = RESULTS_DIR / "qka_target_controls" / "aggregate"
INPUT_CSV = AGGREGATE_DIR / "target_control_mean_std.csv"
OUTPUT_PDF = AGGREGATE_DIR / "target_control_comparison.pdf"
OUTPUT_PNG = AGGREGATE_DIR / "target_control_comparison.png"

AGGREGATE_DIR.mkdir(parents=True, exist_ok=True)

TARGET_ORDER = ["Soft", "Hard", "Randomized"]

TARGET_COLORS = {
    "Soft": "#E2B35D",        # pastel gold
    "Hard": "#7FA9AA",        # pastel teal
    "Randomized": "#B7A6C9",  # pastel purple
}

TARGET_MARKERS = {
    "Soft": "o",
    "Hard": "s",
    "Randomized": "^",
}


def normalize_target_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize possible target-name variants."""
    mapping = {
        "soft": "Soft",
        "soft_target": "Soft",
        "hard": "Hard",
        "hard_target": "Hard",
        "randomized": "Randomized",
        "random": "Randomized",
        "randomized_target": "Randomized",
    }

    df = df.copy()
    df["target"] = (
        df["target"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(mapping)
        .fillna(df["target"])
    )
    return df


def find_column(df: pd.DataFrame, candidates: list[str]) -> str:
    """Return the first matching column name."""
    for column in candidates:
        if column in df.columns:
            return column

    raise ValueError(
        f"Could not find any of these columns: {candidates}\n"
        f"Available columns: {list(df.columns)}"
    )


def plot_target_control_bar(
    df: pd.DataFrame,
    macro_f1_column: str,
) -> None:
    """Plot mean test Macro-F1 with one-standard-deviation error bars."""

    summary = (
        df.groupby("target", as_index=False)
        .agg(
            mean_macro_f1=(macro_f1_column, "mean"),
            sd_macro_f1=(macro_f1_column, "std"),
        )
        .set_index("target")
        .reindex(TARGET_ORDER)
    )

    means = summary["mean_macro_f1"].to_numpy()
    sds = summary["sd_macro_f1"].to_numpy()
    x = np.arange(len(TARGET_ORDER))

    fig, ax = plt.subplots(figsize=(6.5, 4.8))

    bars = ax.bar(
        x,
        means,
        yerr=sds,
        capsize=6,
        color=[TARGET_COLORS[target] for target in TARGET_ORDER],
        edgecolor="black",
        linewidth=0.7,
        width=0.62,
    )

    for bar, mean in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            mean + 0.006,
            f"{mean:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(TARGET_ORDER, fontsize=11)
    ax.set_ylabel(
        r"Test Macro-$F_1$",
        fontsize=13,
     #   fontweight="bold",
    )

    lower_limit = max(0.0, np.nanmin(means - sds) - 0.03)
    upper_limit = np.nanmax(means + sds) + 0.035
    ax.set_ylim(lower_limit, upper_limit)

    ax.grid(
        axis="y",
        linestyle="--",
        linewidth=0.6,
        alpha=0.4,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / "target_control_macro_f1.pdf",
        bbox_inches="tight",
    )
    fig.savefig(
        OUTPUT_DIR / "target_control_macro_f1.png",
        dpi=600,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_alignment_vs_performance(
    df: pd.DataFrame,
    macro_f1_column: str,
    alignment_column: str,
) -> None:
    """Plot final alignment against held-out test Macro-F1."""

    fig, ax = plt.subplots(figsize=(6.6, 4.9))

    for target in TARGET_ORDER:
        target_df = df[df["target"] == target].copy()

        ax.scatter(
            target_df[alignment_column],
            target_df[macro_f1_column],
            color=TARGET_COLORS[target],
            marker=TARGET_MARKERS[target],
            s=70,
            edgecolors="black",
            linewidths=0.6,
            alpha=0.9,
            label=target,
            zorder=3,
        )

        # Connect only the target mean, not individual observations.
        mean_alignment = target_df[alignment_column].mean()
        mean_macro_f1 = target_df[macro_f1_column].mean()

        ax.scatter(
            mean_alignment,
            mean_macro_f1,
            color=TARGET_COLORS[target],
            marker=TARGET_MARKERS[target],
            s=150,
            edgecolors="black",
            linewidths=1.2,
            zorder=4,
        )

    ax.set_xlabel(
        "Final kernel-target alignment",
        fontsize=13,
      #  fontweight="bold",
    )
    ax.set_ylabel(
        r"Test Macro-$F_1$",
        fontsize=13,
      #  fontweight="bold",
    )

    ax.grid(
        linestyle="--",
        linewidth=0.6,
        alpha=0.35,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        frameon=False,
        fontsize=10,
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / "alignment_vs_test_macro_f1.pdf",
        bbox_inches="tight",
    )
    fig.savefig(
        OUTPUT_DIR / "alignment_vs_test_macro_f1.png",
        dpi=600,
        bbox_inches="tight",
    )

    plt.close(fig)


def main() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_CSV}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_CSV)

    target_column = find_column(
      df,
     ["target", "target_mode", "target_type", "alignment_target"],
    )
    macro_f1_column = find_column(
        df,
        ["test_macro_f1", "macro_f1", "test_f1_macro"],
    )

    alignment_column = find_column(
        df,
        [
            "final_training_alignment",
            "final_alignment",
            "alignment_final",
            "best_alignment",
            "alignment_current",
        ],
    )

    if target_column != "target":
        df = df.rename(columns={target_column: "target"})

    df = normalize_target_names(df)

    missing_targets = [
        target for target in TARGET_ORDER
        if target not in set(df["target"])
    ]
    if missing_targets:
        raise ValueError(
            f"Missing target groups: {missing_targets}\n"
            f"Found targets: {sorted(df['target'].unique())}"
        )

    plot_target_control_bar(
        df=df,
        macro_f1_column=macro_f1_column,
    )

    plot_alignment_vs_performance(
        df=df,
        macro_f1_column=macro_f1_column,
        alignment_column=alignment_column,
    )

    print("Created:")
    print(OUTPUT_DIR / "target_control_macro_f1.pdf")
    print(OUTPUT_DIR / "target_control_macro_f1.png")
    print(OUTPUT_DIR / "alignment_vs_test_macro_f1.pdf")
    print(OUTPUT_DIR / "alignment_vs_test_macro_f1.png")


if __name__ == "__main__":
    main()
