#!/usr/bin/env python3

from pathlib import Path
import sys

from src.paths import RESULTS_DIR
import matplotlib.pyplot as plt
import pandas as pd


#!/usr/bin/env python3



from pathlib import Path

import sys



import matplotlib.pyplot as plt

import pandas as pd



REPO_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(REPO_ROOT))



from src.paths import RESULTS_DIR

AGGREGATE_DIR = RESULTS_DIR / "multiseed_matched" / "aggregate"

INPUT_CSV = AGGREGATE_DIR / "all_seed_level_results.csv"

OUTPUT_PDF = AGGREGATE_DIR / "five_seed_macro_f1_comparison.pdf"

OUTPUT_PNG = AGGREGATE_DIR / "five_seed_macro_f1_comparison.png"

MODEL_ORDER = [
    "RBF-SVM",
    "Fixed QK",
    "Gamma-only QKA",
]

DISPLAY_NAMES = {
    "RBF-SVM": "RBF-SVM",
    "Fixed QK": "Fixed QK + SVC",
    "Gamma-only QKA":r"$\gamma$-only QKA + SVC",
}


MODEL_COLORS = {
    "RBF-SVM": "#355F86",
    "Fixed QK": "#7FA9AA",
    "Gamma-only QKA": "#E2B35D",
}

MODEL_MARKERS = {
    "RBF-SVM": "o",
    "Fixed QK": "s",
    "Gamma-only QKA": "^",
}

TRAIN_SIZES = [100, 150, 210]


def main() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)

    required = {
        "model",
        "train_per_group",
        "test_macro_f1",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    summary = (
        df[
            df["model"].isin(MODEL_ORDER)
            & df["train_per_group"].isin(TRAIN_SIZES)
        ]
        .groupby(["train_per_group", "model"], as_index=False)
        .agg(
            mean_macro_f1=("test_macro_f1", "mean"),
            sd_macro_f1=("test_macro_f1", "std"),
        )
    )

    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    for model in MODEL_ORDER:
        model_df = (
            summary[summary["model"] == model]
            .set_index("train_per_group")
            .reindex(TRAIN_SIZES)
        )

        means = model_df["mean_macro_f1"].to_numpy()
        sds = model_df["sd_macro_f1"].to_numpy()

        ax.errorbar(
            TRAIN_SIZES,
            means,
            yerr=sds,
            color=MODEL_COLORS[model],
            marker=MODEL_MARKERS[model],
            linewidth=1.8,
            markersize=6.5,
            capsize=5,
            capthick=1.2,
            elinewidth=1.2,
            markeredgecolor="black",
            markeredgewidth=0.5,
            label=DISPLAY_NAMES[model],
            zorder=3,
        )

        for x_value, y_value in zip(TRAIN_SIZES, means):
            if model == "RBF-SVM":
                y_offset = 0.010
            elif model == "Fixed QK":
                y_offset = -0.014
            else:
                y_offset = 0.010

            ax.text(
                x_value,
                y_value + y_offset,
                f"{y_value:.3f}",
                ha="center",
                va="center",
                fontsize=9,
            )

    ax.set_xlabel(
        "Training samples per group",
        fontsize=14,
      #  fontweight="bold",
    )
    ax.set_ylabel(
        r"Test Macro-$F_1$",
        fontsize=14,
      #  fontweight="bold",
    )

    ax.set_xticks(TRAIN_SIZES)
    ax.set_xlim(95, 215)
    ax.set_ylim(0.48, 0.70)

    ax.tick_params(axis="both", labelsize=11)

    ax.grid(
        axis="y",
        linestyle="--",
        linewidth=0.6,
        alpha=0.45,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        frameon=False,
        fontsize=10,
        ncol=1,
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_PDF,
        bbox_inches="tight",
    )

    fig.savefig(
        OUTPUT_PNG,
        dpi=600,
        bbox_inches="tight",
    )

    print(f"Saved PDF: {OUTPUT_PDF}")
    print(f"Saved PNG: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
