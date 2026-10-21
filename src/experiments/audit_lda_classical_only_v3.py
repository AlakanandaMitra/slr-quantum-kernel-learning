#!/usr/bin/env python3
"""Matched classical-only LDA provenance audit; does not modify existing results."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import SVC

C_GRID = (0.1, 1.0, 10.0, 100.0)
GAMMA_GRID = ("scale", 0.01, 0.1, 1.0)



def evaluate(xtr, ytr, xva, yva, xte, yte):
    candidates = []
    for c in C_GRID:
        for gamma in GAMMA_GRID:
            clf = SVC(kernel="rbf", C=c, gamma=gamma,
                      class_weight="balanced")
            clf.fit(xtr, ytr)
            pred = clf.predict(xva)
            candidates.append((
                f1_score(yva, pred, average="macro", zero_division=0),
                accuracy_score(yva, pred), c, gamma
            ))

    best = max(candidates, key=lambda x: (x[0], x[1]))
    val_f1, val_acc, best_c, best_gamma = best

    clf = SVC(kernel="rbf", C=best_c, gamma=best_gamma,
              class_weight="balanced")
    clf.fit(xtr, ytr)
    pred = clf.predict(xte)

    return {
        "best_C": best_c,
        "best_gamma": best_gamma,
        "val_macro_f1": val_f1,
        "val_accuracy": val_acc,
        "test_macro_f1": f1_score(
            yte, pred, average="macro", zero_division=0),
        "test_accuracy": accuracy_score(yte, pred)
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", type=Path, default=Path.cwd())
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()
    root = args.project.resolve()
    raw = root / "results/embeddings/efficientnet_b0"
    matched = root / "results/Submission_COMPAG/classical_controlled_qka_matched_xgb_gpu"
    runs = root / "results/Submission_COMPAG/qka_gamma_only"
    out = (args.output or root / "results/IEEE_Access/lda_classical_only_audit").resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"Refusing to overwrite nonempty output directory: {out}")

    original = {}
    for split, stem in (("train", "train"), ("val", "valid"), ("test", "test")):
        original[split] = (np.load(raw / f"{stem}_embeddings.npy", mmap_mode="r"),
                           np.load(raw / f"{stem}_label_idxs.npy"))
    fit_idx = np.load(matched / "train_indices.npy")
    pca = joblib.load(matched / "pca_model_64d.joblib")
    shared_lda = joblib.load(matched / "lda_model_11d.joblib")
    assert len(fit_idx) == len(np.unique(fit_idx)) == 120
    assert np.array_equal(np.sort(np.unique(original["train"][1][fit_idx])),
                          np.arange(12)), "Shared fitting subset lacks one or more original classes"
    assert len(getattr(shared_lda, "classes_", [])) == 12
    xfit_raw = np.asarray(original["train"][0][fit_idx])
    yfit = original["train"][1][fit_idx]
    # Direct LDA is fit on the SAME 120 samples and original 12-class labels as shared SLR.
    direct_lda = LinearDiscriminantAnalysis(n_components=11)
    direct_lda.fit(xfit_raw, yfit)
    dirs = sorted(runs.glob("qka_gamma_only_soft_3group_*_seed*"))
    if len(dirs) != 15:
        raise SystemExit(f"Expected 15 matched runs, found {len(dirs)}; check project path")
    rows = []
    for d in dirs:
        name = d.name
        parts = name.removeprefix("qka_gamma_only_soft_3group_").split("_seed")
        size, seed = int(parts[0]), int(parts[1])
        idx = {split: np.load(d / f"{name}_{split}_indices_full.npy")
               for split in ("train", "val", "test")}
        ys = {split: np.load(d / f"{name}_y_{split}_3group.npy")
              for split in ("train", "val", "test")}
        for split in ("train", "val", "test"):
            assert idx[split].ndim == 1
            assert idx[split].min() >= 0 and idx[split].max() < len(original[split][1])
            assert len(idx[split]) == len(ys[split])
        # Reuse the saved SLR features as a reference; never refit its shared LDA.
        slr = {split: np.load(d / f"{name}_X_{split}_pca64_lda11.npy")
               for split in ("train", "val", "test")}
        raw_subset = {split: np.asarray(original[split][0][idx[split]])
                      for split in ("train", "val", "test")}
        fine_train = original["train"][1][idx["train"]]
        # Use the EXACT saved PCA arrays that generated the published SLR features.
        # Re-transforming raw embeddings through the saved PCA model produces small
        # numerical differences and is not an exact reconstruction of those arrays.
        pca_subset = {
            split: np.asarray(np.load(
                matched / f"{'valid' if split == 'val' else split}_embeddings_pca.npy",
                mmap_mode="r")[idx[split]])
            for split in ("train", "val", "test")
        }
        for split in ("train", "val", "test"):
            expected = shared_lda.transform(pca_subset[split])
            if not np.allclose(expected, slr[split], rtol=1e-6, atol=1e-6):
                raise RuntimeError(
                    f"Saved PCA/SLR feature mismatch: {name} {split}; no results written"
                )
        direct = {split: direct_lda.transform(raw_subset[split])
                  for split in ("train", "val", "test")}
        # Keep the original train-only PCA representation fixed; refit ONLY LDA
        # on the downstream training subset's fine-grained labels.
        if len(np.unique(fine_train)) != 12:
            raise RuntimeError(f"Missing original fine class in {name}; cannot fit 11-D LDA")
        local_lda = LinearDiscriminantAnalysis(n_components=11)
        local_lda.fit(pca_subset["train"], fine_train)
        local = {split: local_lda.transform(pca_subset[split])
                 for split in ("train", "val", "test")}
        overlap = len(np.intersect1d(fit_idx, idx["train"]))
        for label, feature in (("shared_slr", slr), ("direct_lda_120", direct),
                               ("subset_refit_slr", local)):
            scores = evaluate(feature["train"], ys["train"],
                              feature["val"], ys["val"],
                              feature["test"], ys["test"])
            rows.append({"train_per_group": size, "seed": seed, "method": label,
                         "n_train": len(idx["train"]), "n_val": len(idx["val"]),
                         "n_test": len(idx["test"]), "shared_fit_n": len(fit_idx),
                         "shared_fit_overlap_train": overlap,
                         "shared_fit_outside_train": len(fit_idx) - overlap,
                         **scores})
        print(f"Completed {name}: shared fitting overlap {overlap}/120", flush=True)
    out.mkdir(parents=True, exist_ok=False)
    df = pd.DataFrame(rows).sort_values(["train_per_group", "seed", "method"])
    df.to_csv(out / "per_run_results.csv", index=False)
    summary = df.groupby(["train_per_group", "method"])["test_macro_f1"].agg(
        ["mean", "std", "count"]).reset_index()
    summary.to_csv(out / "summary_macro_f1.csv", index=False)
    (out / "protocol.json").write_text(json.dumps({
        "purpose": "Exploratory classical-only downstream RBF-SVM comparisons",
        "shared_representation_fit": "PCA64 original training partition; LDA11 120 balanced original 12-class training samples",
        "direct_lda_fit": "Raw 1280-D embeddings of identical 120 fine-label samples",
        "subset_refit": "Reuse exact saved original train-only PCA64 arrays; refit LDA11 on each downstream train subset's original 12-class labels",
        "classifier": "Unscaled RBF SVC, gamma searched over scale, 0.01, 0.1, 1.0; class_weight=balanced; C chosen on validation macro-F1, accuracy tie-break, then first candidate in grid order; train-only final fit",
        "test": "Identical held-out test indices within each seed and training size; test subsets may differ across seeds",
        "limitation": "Shared SLR and direct LDA use 120 representation-fitting labels, some outside downstream training subset; subset-refit LDA uses downstream fine labels; comparisons are conditional, not equal-label-budget end-to-end comparisons",
        "note": "Saved PCA64 arrays reproduce the existing SLR features exactly; raw embeddings transformed anew by the saved PCA model differ numerically and are not substituted. No quantum experiment is executed or modified. Results are exploratory until independently reviewed."
    }, indent=2))
    print("\nRESULTS:", out)
    print(summary.to_string(index=False))

if __name__ == "__main__":
    main()
