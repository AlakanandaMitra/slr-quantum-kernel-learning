#!/usr/bin/env python3



"""

Bootstrap stability analysis for held-out Silhouette coefficients.



Reviewer 2, Concern 5.



Representations:

1. Original EfficientNet-B0 embeddings

2. PCA64

3. PCA64 -> LDA11 (SLR)

4. Direct LDA11 diagnostic reference



The script first verifies that the observed Silhouette coefficients

reproduce the manuscript values before running the bootstrap.



Bootstrap:

- 1,000 replicates

- stratified by the original 12 fine-grained classes

- original test class counts preserved

- same bootstrap indices used for every representation

- transformations are NOT refitted during bootstrap

- paired SLR - PCA64 difference is reported

"""



from pathlib import Path



import joblib

import numpy as np

import pandas as pd



from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.metrics import silhouette_score


import sys

from pathlib import Path



REPO_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(REPO_ROOT))



from src.paths import EMBEDDINGS_DIR, RESULTS_DIR


# ============================================================

# CONFIGURATION

# ============================================================



EMBEDDING_DIR = EMBEDDINGS_DIR / "efficientnet_b0"



LATENT_DIR = RESULTS_DIR / "latent_geometry"



OUTPUT_DIR = RESULTS_DIR / "silhouette_bootstrap"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)



N_BOOTSTRAP = 1000

BOOTSTRAP_SEED = 2026





# ============================================================

# RAW EMBEDDING PATHS

# ============================================================



TRAIN_EMBEDDINGS_PATH = (

    EMBEDDING_DIR

    / "train_embeddings.npy"

)



TEST_EMBEDDINGS_PATH = (

    EMBEDDING_DIR

    / "test_embeddings.npy"

)



TRAIN_LABELS_PATH = (

    EMBEDDING_DIR

    / "train_label_idxs.npy"

)



TEST_LABELS_PATH = (

    EMBEDDING_DIR

    / "test_label_idxs.npy"

)





# ============================================================

# SAVED LATENT-GEOMETRY ARTIFACTS

# ============================================================



PCA64_TEST_PATH = (

    LATENT_DIR

    / "test_embeddings_pca.npy"

)



LDA_MODEL_PATH = (

    LATENT_DIR

    / "lda_model_11d.joblib"

)





# ============================================================

# EXPECTED MANUSCRIPT VALUES

# ============================================================



EXPECTED = {

    "Original": -0.0015,

    "PCA64": -0.0054,

    "SLR_PCA64_LDA11": 0.0490,

    "Direct_LDA11": 0.1444,

}



# Manuscript values are rounded to four decimal places.

TOLERANCE = 0.0001





# ============================================================

# CHECK INPUT FILES

# ============================================================



required_files = [

    TRAIN_EMBEDDINGS_PATH,

    TEST_EMBEDDINGS_PATH,

    TRAIN_LABELS_PATH,

    TEST_LABELS_PATH,

    PCA64_TEST_PATH,

    LDA_MODEL_PATH,

]



print("\nChecking input files...")



for path in required_files:



    if not path.exists():

        raise FileNotFoundError(

            f"Required file not found:\n{path}"

        )



    print(f"  FOUND: {path}")





# ============================================================

# LOAD ORIGINAL DATA

# ============================================================



print("\nLoading original EfficientNet-B0 embeddings...")



X_train = np.load(TRAIN_EMBEDDINGS_PATH)

X_test = np.load(TEST_EMBEDDINGS_PATH)



y_train = np.load(TRAIN_LABELS_PATH).reshape(-1)

y_test = np.load(TEST_LABELS_PATH).reshape(-1)



print(f"X_train: {X_train.shape}")

print(f"X_test : {X_test.shape}")

print(f"y_train: {y_train.shape}")

print(f"y_test : {y_test.shape}")





# ============================================================

# LOAD PCA64 TEST REPRESENTATION

# ============================================================



print("\nLoading saved PCA64 test representation...")



X_test_pca64 = np.load(PCA64_TEST_PATH)



print(f"X_test_pca64: {X_test_pca64.shape}")





# ============================================================

# GENERATE SLR USING SAVED LDA MODEL

# ============================================================



print("\nLoading saved PCA64 -> LDA11 model...")



lda_after_pca = joblib.load(LDA_MODEL_PATH)



X_test_slr = lda_after_pca.transform(

    X_test_pca64

)



print(f"X_test_slr: {X_test_slr.shape}")





# ============================================================

# DIRECT LDA11

#

# Reproduce the diagnostic procedure from latent_geometry.ipynb:

#

# lda_only = LinearDiscriminantAnalysis(n_components=11)

# lda_only.fit(X_train, y_train)

# X_test_lda_only = lda_only.transform(X_test)

# ============================================================



print("\nFitting direct LDA11 diagnostic reference...")



lda_only = LinearDiscriminantAnalysis(

    n_components=11

)



lda_only.fit(

    X_train,

    y_train

)



X_test_lda_only = lda_only.transform(

    X_test

)



print(

    f"X_test_lda_only: {X_test_lda_only.shape}"

)





# ============================================================

# BASIC CONSISTENCY CHECKS

# ============================================================



n_test = len(y_test)



representations = {

    "Original": X_test,

    "PCA64": X_test_pca64,

    "SLR_PCA64_LDA11": X_test_slr,

    "Direct_LDA11": X_test_lda_only,

}



for name, X in representations.items():



    if X.shape[0] != n_test:

        raise ValueError(

            f"{name}: {X.shape[0]} samples, "

            f"but y_test has {n_test}."

        )





classes, counts = np.unique(

    y_test,

    return_counts=True

)



print("\nHeld-out test class counts:")



for c, count in zip(classes, counts):

    print(f"  Class {c}: {count}")



if len(classes) != 12:

    raise ValueError(

        f"Expected 12 fine-grained classes, "

        f"found {len(classes)}."

    )





# ============================================================

# OBSERVED SILHOUETTE SCORES

# ============================================================



print("\n==============================================")

print("PROVENANCE CHECK")

print("==============================================")



observed = {}



for name, X in representations.items():



    score = silhouette_score(

        X,

        y_test,

        metric="euclidean"

    )



    observed[name] = score



    print(

        f"{name:22s}: "

        f"{score: .6f} "

        f"(manuscript: {EXPECTED[name]: .4f})"

    )





# ============================================================

# VERIFY AGAINST MANUSCRIPT VALUES

# ============================================================



print("\nChecking agreement with manuscript...")



failed = []



for name in EXPECTED:



    rounded_observed = round(

        observed[name],

        4

    )



    if abs(

        rounded_observed - EXPECTED[name]

    ) > TOLERANCE:



        failed.append(

            (

                name,

                observed[name],

                EXPECTED[name]

            )

        )





if failed:



    print("\nERROR:")

    print(

        "One or more Silhouette values do not "

        "reproduce the manuscript."

    )



    for name, actual, expected in failed:



        print(

            f"  {name}: "

            f"observed={actual:.6f}, "

            f"expected≈{expected:.4f}"

        )



    raise RuntimeError(

        "\nBootstrap NOT performed because the "

        "provenance check failed."

    )





print(

    "\nSUCCESS: manuscript Silhouette values reproduced."

)

print(

    "Proceeding to stratified bootstrap..."

)





# ============================================================

# STRATIFIED BOOTSTRAP

# ============================================================



rng = np.random.default_rng(

    BOOTSTRAP_SEED

)



class_indices = {

    c: np.where(y_test == c)[0]

    for c in classes

}



bootstrap_scores = {

    name: np.empty(

        N_BOOTSTRAP,

        dtype=float

    )

    for name in representations

}



delta_slr_pca64 = np.empty(

    N_BOOTSTRAP,

    dtype=float

)





print(

    f"\nRunning {N_BOOTSTRAP} "

    "stratified bootstrap replicates..."

)





for b in range(N_BOOTSTRAP):



    sampled_parts = []



    # Resample within each fine-grained class.

    # Original class counts are preserved.

    for c in classes:



        idx_c = class_indices[c]



        sampled_c = rng.choice(

            idx_c,

            size=len(idx_c),

            replace=True

        )



        sampled_parts.append(

            sampled_c

        )



    boot_idx = np.concatenate(

        sampled_parts

    )



    y_boot = y_test[boot_idx]



    # Same bootstrap sample for all representations.

    for name, X in representations.items():



        bootstrap_scores[name][b] = (

            silhouette_score(

                X[boot_idx],

                y_boot,

                metric="euclidean"

            )

        )



    delta_slr_pca64[b] = (

        bootstrap_scores[

            "SLR_PCA64_LDA11"

        ][b]

        -

        bootstrap_scores[

            "PCA64"

        ][b]

    )



    if (b + 1) % 100 == 0:



        print(

            f"  Completed "

            f"{b + 1}/{N_BOOTSTRAP}"

        )





# ============================================================

# SUMMARY FUNCTION

# ============================================================



def summarize(values):



    return {

        "Bootstrap_Mean":

            np.mean(values),



        "Bootstrap_SD":

            np.std(

                values,

                ddof=1

            ),



        "CI_2.5":

            np.percentile(

                values,

                2.5

            ),



        "CI_97.5":

            np.percentile(

                values,

                97.5

            ),

    }





# ============================================================

# REPRESENTATION RESULTS

# ============================================================



rows = []



for name, values in bootstrap_scores.items():



    stats = summarize(values)



    rows.append(

        {

            "Representation": name,

            "Observed_Silhouette":

                observed[name],

            **stats

        }

    )





summary_df = pd.DataFrame(

    rows

)





# ============================================================

# PAIRED SLR - PCA64 DIFFERENCE

# ============================================================



delta_stats = summarize(

    delta_slr_pca64

)



observed_delta = (

    observed["SLR_PCA64_LDA11"]

    -

    observed["PCA64"]

)



delta_df = pd.DataFrame(

    [

        {

            "Comparison":

                "SLR minus PCA64",



            "Observed_Difference":

                observed_delta,



            **delta_stats,



            "Fraction_Delta_GT_0":

                np.mean(

                    delta_slr_pca64 > 0

                ),

        }

    ]

)





# ============================================================

# SAVE OUTPUT

# ============================================================



OUTPUT_DIR.mkdir(

    parents=True,

    exist_ok=True

)



summary_df.to_csv(

    OUTPUT_DIR

    / "silhouette_bootstrap_summary.csv",

    index=False

)



delta_df.to_csv(

    OUTPUT_DIR

    / "silhouette_bootstrap_delta_slr_pca64.csv",

    index=False

)



for name, values in bootstrap_scores.items():



    np.save(

        OUTPUT_DIR

        / f"{name}_bootstrap_scores.npy",

        values

    )





np.save(

    OUTPUT_DIR

    / "SLR_minus_PCA64_bootstrap_delta.npy",

    delta_slr_pca64

)





# ============================================================

# FINAL OUTPUT

# ============================================================



print("\n")

print("=" * 78)

print("BOOTSTRAP SILHOUETTE RESULTS")

print("=" * 78)



print(

    summary_df.to_string(

        index=False,

        float_format=lambda x: f"{x:.6f}"

    )

)





print("\n")

print("=" * 78)

print("PAIRED SLR - PCA64 DIFFERENCE")

print("=" * 78)



print(

    delta_df.to_string(

        index=False,

        float_format=lambda x: f"{x:.6f}"

    )

)





print("\nResults saved to:")

print(OUTPUT_DIR)



print("\nAnalysis complete.")
