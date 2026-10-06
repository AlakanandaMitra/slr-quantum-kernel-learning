

#!/usr/bin/env python3

"""PCA-dimensionality sensitivity using saved IEEE Access matched splits."""



import argparse

import json

import re

from pathlib import Path



import numpy as np

import pandas as pd

from sklearn.decomposition import PCA

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.metrics import accuracy_score, f1_score

from sklearn.svm import SVC



C_GRID = (0.1, 1.0, 10.0, 100.0)

GAMMA_GRID = ("scale", 0.01, 0.1, 1.0)

DIMENSIONS = (16, 32, 64, 128)

TRAIN_SIZES = (100, 150, 210)





def load(path):

    if not path.is_file():

        raise FileNotFoundError(path)

    return np.load(path, allow_pickle=False)





def check_indices(indices, n_rows, description):

    if (

        indices.ndim != 1

        or not np.issubdtype(indices.dtype, np.integer)

        or len(indices) == 0

        or indices.min() < 0

        or indices.max() >= n_rows

        or len(np.unique(indices)) != len(indices)

    ):

        raise ValueError(f"Invalid indices: {description}")





def discover_runs(gamma_root):

    pattern = re.compile(

        r"^qka_gamma_only_soft_3group_(100|150|210)_seed(\d+)$"

    )

    runs = []



    for folder in sorted(gamma_root.iterdir()):

        if not folder.is_dir():

            continue

        match = pattern.fullmatch(folder.name)

        if match:

            size, seed = map(int, match.groups())

            runs.append((size, seed, folder))



    for size in TRAIN_SIZES:

        seeds = [seed for n, seed, _ in runs if n == size]

        if len(seeds) != 5 or len(set(seeds)) != 5:

            raise ValueError(

                f"Expected five distinct saved seeds for "

                f"{size}/group; found {seeds}"

            )



    return runs





def load_saved_run(size, seed, folder, lengths):

    name = folder.name

    indices = {}

    labels = {}



    for part in ("train", "val", "test"):

        indices[part] = load(

            folder / f"{name}_{part}_indices_full.npy"

        )

        labels[part] = load(

            folder / f"{name}_y_{part}_3group.npy"

        )



        check_indices(

            indices[part], lengths[part], f"{name}/{part}"

        )



        if len(indices[part]) != len(labels[part]):

            raise ValueError(f"Index/label mismatch: {name}/{part}")



        if not set(np.unique(labels[part])).issubset({0, 1, 2}):

            raise ValueError(f"Unexpected group labels: {name}/{part}")



    if len(indices["train"]) != 3 * size:

        raise ValueError(f"Unexpected training size: {name}")



    return {

        "size": size,

        "seed": seed,

        "indices": indices,

        "labels": labels,

    }





def evaluate_rbf(features, run):

    indices = run["indices"]

    labels = run["labels"]



    x_train = features["train"][indices["train"]]

    x_val = features["val"][indices["val"]]

    x_test = features["test"][indices["test"]]



    y_train = labels["train"]

    y_val = labels["val"]

    y_test = labels["test"]



    candidates = []



    for C in C_GRID:

        for gamma in GAMMA_GRID:

            model = SVC(

                kernel="rbf",

                C=C,

                gamma=gamma,

                class_weight="balanced",

            )

            model.fit(x_train, y_train)

            predictions = model.predict(x_val)



            candidates.append({

                "C": C,

                "gamma": gamma,

                "val_macro_f1": f1_score(

                    y_val,

                    predictions,

                    average="macro",

                    zero_division=0,

                ),

                "val_accuracy": accuracy_score(

                    y_val, predictions

                ),

            })



    best = max(

        candidates,

        key=lambda item: (

            item["val_macro_f1"],

            item["val_accuracy"],

        ),

    )



    final_model = SVC(

        kernel="rbf",

        C=best["C"],

        gamma=best["gamma"],

        class_weight="balanced",

    )



    # Match the original baseline: train only, not train + validation.

    final_model.fit(x_train, y_train)

    predictions = final_model.predict(x_test)



    return {

        "train_per_group": run["size"],

        "seed": run["seed"],

        "selected_C": best["C"],

        "selected_gamma": best["gamma"],

        "val_macro_f1": best["val_macro_f1"],

        "test_macro_f1": f1_score(

            y_test,

            predictions,

            average="macro",

            zero_division=0,

        ),

        "test_accuracy": accuracy_score(y_test, predictions),

        "num_train": len(y_train),

        "num_val": len(y_val),

        "num_test": len(y_test),

    }





def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(

        "--raw-dir",

        type=Path,

        default=Path("results/embeddings/efficientnet_b0"),

    )

    parser.add_argument(

        "--matched-dir",

        type=Path,

        default=Path(

            "results/"

            "classical_controlled_qka_matched_xgb_gpu"

        ),

    )

    parser.add_argument(

        "--gamma-root",

        type=Path,

        default=Path("results/qka_gamma_only"),

    )

    parser.add_argument(

        "--output-dir",

        type=Path,

        default=Path(

            "results/pca_dim_sensitivity"

        ),

    )

    args = parser.parse_args()



    raw = {

        "train": load(args.raw_dir / "train_embeddings.npy"),

        "val": load(args.raw_dir / "valid_embeddings.npy"),

        "test": load(args.raw_dir / "test_embeddings.npy"),

    }

    fine_labels = load(args.raw_dir / "train_label_idxs.npy")

    lda_indices = load(args.matched_dir / "train_indices.npy")



    check_indices(

        lda_indices, len(raw["train"]), "LDA fitting indices"

    )



    fine_classes, counts = np.unique(

        fine_labels[lda_indices], return_counts=True

    )

    if (

        len(lda_indices) != 120

        or len(fine_classes) != 12

        or not np.all(counts == 10)

    ):

        raise ValueError(

            "Expected 120 LDA-fitting samples: "

            "10 from each of 12 fine-grained classes."

        )



    lengths = {part: len(x) for part, x in raw.items()}

    runs = [

        load_saved_run(size, seed, folder, lengths)

        for size, seed, folder in discover_runs(args.gamma_root)

    ]



    args.output_dir.mkdir(parents=True, exist_ok=True)



    protocol = {

        "dimensions": DIMENSIONS,

        "pca_fit": "original training embeddings only",

        "pca_random_state": 42,

        "lda_fit": "same saved 120 indices and 12-class labels",

        "lda_components": 11,

        "downstream_splits": "saved gamma-only full indices",

        "classifier": "unscaled LDA features; balanced RBF-SVM",

        "C_grid": C_GRID,

        "gamma_grid": GAMMA_GRID,

        "selection": "validation macro-F1, then accuracy",

        "final_fit": "training split only",

        "note": (

            "All PCA dimensions, including 64, are refitted. "

            "Original saved PCA64 features are not mixed into "

            "the primary sensitivity comparison."

        ),

    }

    (args.output_dir / "protocol.json").write_text(

        json.dumps(protocol, indent=2)

    )



    results = []



    for dimension in DIMENSIONS:

        print(f"\nFitting PCA{dimension}...", flush=True)



        pca = PCA(n_components=dimension, random_state=42)

        reduced = {

            "train": pca.fit_transform(raw["train"])

        }

        reduced["val"] = pca.transform(raw["val"])

        reduced["test"] = pca.transform(raw["test"])



        lda = LinearDiscriminantAnalysis(n_components=11)

        lda.fit(

            reduced["train"][lda_indices],

            fine_labels[lda_indices],

        )



        features = {

            part: lda.transform(reduced[part])

            for part in ("train", "val", "test")

        }



        explained_variance = float(

            pca.explained_variance_ratio_.sum()

        )



        for run in runs:

            metrics = evaluate_rbf(features, run)

            metrics["pca_dim"] = dimension

            metrics["lda_dim"] = 11

            metrics["pca_explained_variance"] = explained_variance

            results.append(metrics)



            print(

                f"PCA{dimension} | "

                f"{run['size']}/group | "

                f"seed {run['seed']} | "

                f"macro-F1 {metrics['test_macro_f1']:.4f}",

                flush=True,

            )



        pd.DataFrame(results).to_csv(

            args.output_dir / "per_run.csv", index=False

        )



    df = pd.DataFrame(results)

    summary = (

        df.groupby(["pca_dim", "train_per_group"], as_index=False)

        .agg(

            n_seeds=("test_macro_f1", "count"),

            macro_f1_mean=("test_macro_f1", "mean"),

            macro_f1_sd=("test_macro_f1", "std"),

            explained_variance=(

                "pca_explained_variance", "first"

            ),

        )

    )



    summary.to_csv(

        args.output_dir / "summary.csv", index=False

    )

    print("\nFINAL SUMMARY")

    print(summary.to_string(index=False))

    print(f"\nSaved results to: {args.output_dir}")





if __name__ == "__main__":

    main()
