from pathlib import Path

import os

import sys

import pandas as pd

import numpy as np



REPO_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(REPO_ROOT))



from src.paths import RESULTS_DIR
from src.classical import run_matched_classical_12class_50 as original


SEEDS = [42, 52, 62, 72, 82]



QK_BASE = RESULTS_DIR / "fixed_qk_12class_5seed"



OUTPUT_BASE = RESULTS_DIR / "matched_classical_12class_5seed"



OUTPUT_BASE.mkdir(parents=True, exist_ok=True)




def make_compatibility_split(seed):



    source_dir = QK_BASE / f"seed_{seed}"



    source_prefix = f"fixed_qk_12class_50_seed{seed}"



    compat_dir = (

        OUTPUT_BASE

        / "_split_compatibility"

        / f"seed_{seed}"

    )



    compat_dir.mkdir(parents=True, exist_ok=True)



    mappings = {

        "X_train_pca64_lda11.npy":

            "X_train_pca64_lda11.npy",



        "X_val_pca64_lda11.npy":

            "X_val_pca64_lda11.npy",



        "X_test_pca64_lda11.npy":

            "X_test_pca64_lda11.npy",



        "y_train_12class.npy":

            "y_train_12class.npy",



        "y_val_12class.npy":

            "y_val_12class.npy",



        "y_test_12class.npy":

            "y_test_12class.npy",

    }



    for source_suffix, target_suffix in mappings.items():



        source = (

            source_dir

            / f"{source_prefix}_{source_suffix}"

        )



        target = (

            compat_dir

            / f"fixed_qk_12class_50_per_class_{target_suffix}"

        )



        if not source.exists():

            raise FileNotFoundError(

                f"Missing source split file: {source}"

            )



        if target.exists() or target.is_symlink():

            target.unlink()



        os.symlink(source, target)



    return compat_dir





def run_seed(seed):



    print("\n")

    print("=" * 90)

    print(

        f"CLASSICAL 12-CLASS MATCHED EVALUATION | SEED {seed}"

    )

    print("=" * 90)



    compat_dir = make_compatibility_split(seed)



    run_name = (

        f"matched_classical_12class_50_seed{seed}"

    )



    results_dir = (

        OUTPUT_BASE

        / f"seed_{seed}"

    )



    results_dir.mkdir(

        parents=True,

        exist_ok=True

    )



    # Change ONLY seed, split location,

    # run name and output directory.

    original.SEED = seed

    original.SPLIT_DIR = compat_dir

    original.RUN_NAME = run_name

    original.RESULTS_DIR = results_dir



    # Run the ORIGINAL experiment.

    original.main()



    result_file = (

        results_dir

        / f"{run_name}_all_results.csv"

    )



    if not result_file.exists():

        raise FileNotFoundError(

            f"Expected result file not found: {result_file}"

        )



    df = pd.read_csv(result_file)



    df.insert(0, "seed", seed)



    return df





def aggregate(all_seed_frames):



    aggregate_dir = OUTPUT_BASE / "aggregate"



    aggregate_dir.mkdir(

        parents=True,

        exist_ok=True

    )



    all_df = pd.concat(

        all_seed_frames,

        ignore_index=True

    )



    all_df.to_csv(

        aggregate_dir

        / "matched_classical_12class_5seed_all_results.csv",

        index=False

    )



    summary_rows = []



    for model_name, group in all_df.groupby("model_name"):



        for metric in [

            "test_accuracy",

            "test_macro_f1",

            "test_weighted_f1",

        ]:



            values = group[metric].to_numpy(dtype=float)



            summary_rows.append({

                "model_name": model_name,

                "metric": metric,

                "n": len(values),

                "mean": np.mean(values),

                "std": np.std(values, ddof=1),

                "min": np.min(values),

                "max": np.max(values),

            })



    summary_df = pd.DataFrame(summary_rows)



    summary_df.to_csv(

        aggregate_dir

        / "matched_classical_12class_5seed_mean_std.csv",

        index=False

    )



    print("\n")

    print("=" * 90)

    print("FIVE-SEED CLASSICAL RESULTS — MACRO-F1")

    print("=" * 90)



    pivot = all_df.pivot(

        index="seed",

        columns="model_name",

        values="test_macro_f1",

    )



    print(pivot.to_string())



    print("\n")

    print("=" * 90)

    print("FIVE-SEED CLASSICAL SUMMARY")

    print("=" * 90)



    macro = summary_df[

        summary_df["metric"] == "test_macro_f1"

    ].copy()



    macro = macro.sort_values(

        "mean",

        ascending=False

    )



    print(

        macro[

            [

                "model_name",

                "mean",

                "std",

                "min",

                "max",

            ]

        ].to_string(index=False)

    )



    print("\nManuscript-ready Macro-F1:")



    for _, row in macro.iterrows():



        print(

            f"{row['model_name']}: "

            f"{row['mean']:.4f} ± {row['std']:.4f}"

        )





def main():



    print("=" * 90)

    print(

        "Reviewer 1 Concern 4: "

        "Five-seed matched 12-class classical evaluation"

    )

    print("=" * 90)



    print("Seeds:", SEEDS)



    all_seed_frames = []



    for seed in SEEDS:



        df = run_seed(seed)



        all_seed_frames.append(df)



        # Progressive save in case a later seed fails.

        pd.concat(

            all_seed_frames,

            ignore_index=True

        ).to_csv(

            OUTPUT_BASE

            / "completed_seed_results.csv",

            index=False

        )



    aggregate(all_seed_frames)



    print("\nAll five classical seeds completed.")

    print("Results:", OUTPUT_BASE)





if __name__ == "__main__":

    main()
