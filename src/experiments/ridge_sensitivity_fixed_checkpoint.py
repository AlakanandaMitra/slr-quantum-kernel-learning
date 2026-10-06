
#!/usr/bin/env python3

"""Post-training ridge sensitivity for saved soft-target QKA kernels.



Quantum parameters and the originally validation-selected C are held fixed.

Only the diagonal ridge added to the final training kernel is varied.

No test-set results are used for hyperparameter selection.

"""



import csv

import json

from pathlib import Path



import numpy as np

from sklearn.metrics import accuracy_score, f1_score

from sklearn.svm import SVC





ROOT = Path("results/qka_gamma_only")

OUT = Path("results/IEEE_Access/ridge_sensitivity_fixed_checkpoint")

RIDGES = (0.0, 0.01, 0.1, 1.0)

TRAIN_SIZES = (100, 150, 210)

SEEDS = (42, 52, 62, 72, 82)





def load_array(directory, prefix, suffix):

    path = directory / f"{prefix}_{suffix}.npy"

    if not path.is_file():

        raise FileNotFoundError(path)

    return np.load(path, allow_pickle=False)





def main():

    # Validate all inputs before writing results.

    runs = []



    for size in TRAIN_SIZES:

        for seed in SEEDS:

            prefix = f"qka_gamma_only_soft_3group_{size}_seed{seed}"

            directory = ROOT / prefix



            raw = load_array(directory, prefix, "kernel_train_raw_final")

            regularized = load_array(

                directory, prefix, "kernel_train_regularized_final"

            )

            test_train = load_array(

                directory, prefix, "kernel_test_train_final"

            )

            y_train = load_array(directory, prefix, "y_train_3group")

            y_test = load_array(directory, prefix, "y_test_3group")

            original_predictions = load_array(

                directory, prefix, "predictions"

            )



            checkpoint_path = directory / f"{prefix}_best_checkpoint.json"

            with checkpoint_path.open("r") as f:

                checkpoint = json.load(f)

            C = float(checkpoint["best_C"])



            n_train = len(y_train)

            if raw.shape != (n_train, n_train):

                raise ValueError(f"{prefix}: invalid raw kernel shape {raw.shape}")

            if regularized.shape != raw.shape:

                raise ValueError(f"{prefix}: regularized kernel shape mismatch")

            if test_train.shape != (len(y_test), n_train):

                raise ValueError(

                    f"{prefix}: invalid test kernel shape {test_train.shape}"

                )

            if original_predictions.shape != y_test.shape:

                raise ValueError(f"{prefix}: saved prediction shape mismatch")

            if not all(

                np.all(np.isfinite(a))

                for a in (raw, regularized, test_train)

            ):

                raise ValueError(f"{prefix}: non-finite kernel values")

            if not np.allclose(

                regularized,

                raw + np.float32(0.1) * np.eye(n_train, dtype=np.float32),

                rtol=0,

                atol=1e-6,

            ):

                raise ValueError(

                    f"{prefix}: saved regularized kernel is not raw + 0.1I"

                )



            runs.append(

                (size, seed, prefix, raw, test_train,

                 y_train, y_test, original_predictions, C)

            )



    OUT.mkdir(parents=True, exist_ok=True)

    rows = []



    for (size, seed, prefix, raw, test_train,

         y_train, y_test, original_predictions, C) in runs:



        identity = np.eye(len(y_train), dtype=np.float32)



        for ridge in RIDGES:

            # Match the original code: regularize TRAIN only.

            K_train = raw + np.float32(ridge) * identity



            clf = SVC(

                kernel="precomputed",

                C=C,

                class_weight="balanced",

            )

            clf.fit(K_train, y_train)

            predictions = clf.predict(test_train)



            agreement = (

                float(np.mean(predictions == original_predictions))

                if ridge == 0.1 else None

            )



            row = {

                "train_per_group": size,

                "seed": seed,

                "ridge": ridge,

                "fixed_C": C,

                "test_n": len(y_test),

                "test_macro_f1": f1_score(

                    y_test, predictions, average="macro", zero_division=0

                ),

                "test_accuracy": accuracy_score(y_test, predictions),

                "original_prediction_agreement_at_0p1": agreement,

            }

            rows.append(row)



            print(

                f"{prefix} | ridge={ridge:g} | C={C:g} | "

                f"Macro-F1={row['test_macro_f1']:.6f} | "

                f"accuracy={row['test_accuracy']:.6f}"

                + (

                    f" | original agreement={agreement:.6f}"

                    if agreement is not None else ""

                ),

                flush=True,

            )



            if ridge == 0.1 and agreement != 1.0:

                raise RuntimeError(

                    f"{prefix}: ridge=0.1 does not reproduce "

                    "the original saved predictions. "

                    "Investigate before interpreting the results."

                )



    per_run_path = OUT / "per_run_results.csv"

    with per_run_path.open("w", newline="") as f:

        writer = csv.DictWriter(f, fieldnames=list(rows[0]))

        writer.writeheader()

        writer.writerows(rows)



    summary_rows = []

    for size in TRAIN_SIZES:

        for ridge in RIDGES:

            values = np.array(

                [

                    row["test_macro_f1"]

                    for row in rows

                    if row["train_per_group"] == size

                    and row["ridge"] == ridge

                ],

                dtype=float,

            )

            summary_rows.append(

                {

                    "train_per_group": size,

                    "ridge": ridge,

                    "n_seeds": len(values),

                    "test_macro_f1_mean": float(values.mean()),

                    "test_macro_f1_sample_sd": float(values.std(ddof=1)),

                }

            )



    summary_path = OUT / "summary.csv"

    with summary_path.open("w", newline="") as f:

        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0]))

        writer.writeheader()

        writer.writerows(summary_rows)



    print("\nALL 15 ORIGINAL-PREDICTION CHECKS PASSED.")

    print(f"Per-run results: {per_run_path}")

    print(f"Summary:         {summary_path}")

    print("\nMean ± sample SD of test Macro-F1:")

    for row in summary_rows:

        print(

            f"train/group={row['train_per_group']:3d} "

            f"ridge={row['ridge']:4g}: "

            f"{row['test_macro_f1_mean']:.4f} ± "

            f"{row['test_macro_f1_sample_sd']:.4f}"

        )





if __name__ == "__main__":

    main()

