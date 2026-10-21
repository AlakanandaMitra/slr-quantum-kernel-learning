#!/usr/bin/env python3
"""Matched seed-42 direct-LDA versus shared-SLR fixed-QK diagnostic."""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_matched_multiseed_baselines import (
    load_saved_split,
    run_fixed_qk,
)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", type=Path, default=Path.cwd())
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    root = args.project.resolve()
    out = args.output.resolve()

    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"Refusing to overwrite: {out}")

    n, seed = 150, 42
    runs = root / "results/Submission_COMPAG/qka_gamma_only"
    raw = root / "results/embeddings/efficientnet_b0"
    matched = root / (
        "results/Submission_COMPAG/"
        "classical_controlled_qka_matched_xgb_gpu"
    )

    run_dir, name, shared = load_saved_split(runs, n, seed)

    fit_idx = np.load(matched / "train_indices.npy")
    Xfit = np.load(raw / "train_embeddings.npy", mmap_mode="r")[fit_idx]
    yfit = np.load(raw / "train_label_idxs.npy")[fit_idx]

    assert len(fit_idx) == 120
    assert len(np.unique(yfit)) == 12

    lda = LinearDiscriminantAnalysis(n_components=11)
    lda.fit(np.asarray(Xfit), yfit)

    direct = {}
    for split, stem in (
        ("train", "train"),
        ("val", "valid"),
        ("test", "test")
    ):
        idx = np.load(
            run_dir / f"{name}_{split}_indices_full.npy"
        )

        Xraw = np.load(
            raw / f"{stem}_embeddings.npy", mmap_mode="r"
        )[idx]

        direct[f"X_{split}"] = lda.transform(
            np.asarray(Xraw)
        )
        direct[f"y_{split}"] = shared[f"y_{split}"]

        assert len(direct[f"X_{split}"]) == len(shared[f"X_{split}"])
        assert direct[f"X_{split}"].shape[1] == 11

    out.mkdir(parents=True, exist_ok=True)

    results = {}

    for label, arrays in (
        ("shared_slr", shared),
        ("direct_lda", direct)
    ):
        folder = out / label
        folder.mkdir()

        print(f"Running fixed QK: {label}", flush=True)

        metrics = run_fixed_qk(
            arrays, folder, n, seed,
            device="GPU",
            precision="single"
        )
        metrics["diagnostic_representation"] = label
        results[label] = metrics

    reference = json.loads((
        root / "results/Submission_COMPAG/multiseed_matched/"
        "train150_seed42/"
        "fixed_qk_3group_150_seed42_results_summary.json"
    ).read_text())

    expected = reference["test_macro_f1"]
    actual = results["shared_slr"]["test_macro_f1"]

    # Require exact agreement to numerical tolerance.
    passed = abs(actual - expected) <= 1e-7

    summary = {
        "training_samples_per_group": n,
        "seed": seed,
        "shared_slr": results["shared_slr"],
        "direct_lda": results["direct_lda"],
        "reference_test_macro_f1": expected,
        "shared_slr_reproduction_pass": passed,
        "macro_f1_difference_direct_minus_shared": (
            results["direct_lda"]["test_macro_f1"] - actual
        ),
        "note": (
            "Representative matched fixed-QK comparison; "
            "same downstream splits and 120 LDA-fitting samples; "
            "not a complete equal-data-budget comparison."
        )
    }

    (out / "diagnostic_summary.json").write_text(
        json.dumps(summary, indent=2)
    )

    print("\nShared SLR:", actual)
    print("Original reference:", expected)
    print("Reference reproduction:", "PASS" if passed else "FAIL")
    print(
        "Direct LDA:",
        results["direct_lda"]["test_macro_f1"]
    )
    print(
        "Direct minus shared:",
        summary["macro_f1_difference_direct_minus_shared"]
    )

    if not passed:
        raise SystemExit(
            "Shared-SLR reference reproduction failed; "
            "do not interpret direct-LDA comparison."
        )

if __name__ == "__main__":
    main()
