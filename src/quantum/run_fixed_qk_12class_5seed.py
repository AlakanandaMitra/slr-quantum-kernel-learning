from __future__ import annotations



import json

import platform

import subprocess

import time

import warnings

from pathlib import Path



import sys



REPO_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(REPO_ROOT))



from src.paths import EMBEDDINGS_DIR, RESULTS_DIR



from typing import Tuple, Dict, Any



import joblib

import numpy as np

import pandas as pd



from sklearn.metrics import (

    accuracy_score,

    f1_score,

    classification_report,

    confusion_matrix,

)

from sklearn.svm import SVC



warnings.filterwarnings("ignore")





# ============================================================

# 1. CONFIG

# ============================================================



SEEDS = [42, 52, 62, 72, 82]



MATCHED_DIR = RESULTS_DIR / "matched_representation"



RAW_LABEL_DIR = EMBEDDINGS_DIR / "efficientnet_b0"



BASE_RESULTS_DIR = RESULTS_DIR / "fixed_qk_12class_5seed"



BASE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)



N_CLASSES = 12

N_QUBITS = 11



TRAIN_PER_CLASS = 50

VAL_PER_CLASS = 3



FEATURE_MAP_REPS = 3

ENTANGLEMENT = "circular"



USE_AER = True

USE_AER_GPU = True

AER_PRECISION = "single"



KERNEL_RIDGE = 0.1



SVC_C_GRID = [0.1, 1.0, 10.0, 100.0]



CLASS_NAMES = {

    0: "complex",

    1: "frog_eye_leaf_spot",

    2: "frog_eye_leaf_spot_complex",

    3: "healthy",

    4: "powdery_mildew",

    5: "powdery_mildew_complex",

    6: "rust",

    7: "rust_complex",

    8: "rust_frog_eye_leaf_spot",

    9: "scab",

    10: "scab_frog_eye_leaf_spot",

    11: "scab_frog_eye_leaf_spot_complex",

}





# ============================================================

# 2. SYSTEM INFO

# ============================================================



def get_nvidia_smi_text() -> str:

    try:

        result = subprocess.run(

            ["nvidia-smi"],

            stdout=subprocess.PIPE,

            stderr=subprocess.PIPE,

            text=True,

            check=False,

        )



        if result.returncode == 0:

            return result.stdout



        return (

            f"nvidia-smi returned code {result.returncode}\n"

            f"STDERR:\n{result.stderr}"

        )



    except Exception as e:

        return f"Could not run nvidia-smi: {repr(e)}"





def save_system_info(seed: int, run_name: str, results_dir: Path) -> None:



    info = {

        "platform": platform.platform(),

        "python_version": platform.python_version(),

        "seed": seed,

        "train_sampling_seed": seed,

        "validation_sampling_seed": seed + 1,

        "test_order_seed": seed + 2,

        "run_name": run_name,

        "task": (

            "12-class fixed quantum kernel, "

            "50 train samples per class, five-seed evaluation"

        ),

        "matched_dir": str(MATCHED_DIR),

        "raw_label_dir": str(RAW_LABEL_DIR),

        "results_dir": str(results_dir),

        "n_classes": N_CLASSES,

        "n_qubits": N_QUBITS,

        "train_per_class": TRAIN_PER_CLASS,

        "val_per_class": VAL_PER_CLASS,

        "test_protocol": "all available held-out test samples",

        "representation": "PCA64 -> LDA11",

        "encoding": "[0,1] train-only min-max scaling",

        "feature_map": "ZZFeatureMap(x)",

        "feature_map_reps": FEATURE_MAP_REPS,

        "entanglement": ENTANGLEMENT,

        "use_aer_gpu": USE_AER_GPU,

        "aer_precision": AER_PRECISION,

        "kernel_ridge": KERNEL_RIDGE,

        "svc_c_grid": SVC_C_GRID,

    }



    with open(

        results_dir / f"{run_name}_system_info.json",

        "w"

    ) as f:

        json.dump(info, f, indent=2)



    with open(

        results_dir / f"{run_name}_nvidia_smi.txt",

        "w"

    ) as f:

        f.write(get_nvidia_smi_text())





# ============================================================

# 3. SAMPLING

# ============================================================



def sample_exact_per_class(

    y: np.ndarray,

    per_class: int,

    seed: int,

    split_name: str,

) -> np.ndarray:



    rng_local = np.random.default_rng(seed)



    selected = []



    for c in range(N_CLASSES):



        idx = np.where(y == c)[0].copy()



        rng_local.shuffle(idx)



        if len(idx) < per_class:

            raise ValueError(

                f"{split_name}: class {c} has only {len(idx)} samples, "

                f"but {per_class} requested."

            )



        selected.extend(idx[:per_class])



    selected = np.array(selected, dtype=np.int64)



    rng_local.shuffle(selected)



    return selected





def use_all_indices(

    y: np.ndarray,

    seed: int

) -> np.ndarray:



    rng_local = np.random.default_rng(seed)



    idx = np.arange(

        len(y),

        dtype=np.int64

    )



    rng_local.shuffle(idx)



    return idx





# ============================================================

# 4. LOAD FULL PCA64 -> LDA11 REPRESENTATION ONCE

# ============================================================



def load_full_pca64_lda11_data():



    required = [

        MATCHED_DIR / "train_embeddings_pca.npy",

        MATCHED_DIR / "valid_embeddings_pca.npy",

        MATCHED_DIR / "test_embeddings_pca.npy",

        MATCHED_DIR / "lda_model_11d.joblib",

        RAW_LABEL_DIR / "train_label_idxs.npy",

        RAW_LABEL_DIR / "valid_label_idxs.npy",

        RAW_LABEL_DIR / "test_label_idxs.npy",

    ]



    missing = [

        str(p)

        for p in required

        if not p.exists()

    ]



    if missing:

        raise FileNotFoundError(

            "Missing required files:\n" +

            "\n".join(missing)

        )



    X_train_pca64_full = np.load(

        MATCHED_DIR / "train_embeddings_pca.npy"

    )



    X_val_pca64_full = np.load(

        MATCHED_DIR / "valid_embeddings_pca.npy"

    )



    X_test_pca64_full = np.load(

        MATCHED_DIR / "test_embeddings_pca.npy"

    )



    y_train_full = np.load(

        RAW_LABEL_DIR / "train_label_idxs.npy"

    ).astype(np.int64)



    y_val_full = np.load(

        RAW_LABEL_DIR / "valid_label_idxs.npy"

    ).astype(np.int64)



    y_test_full = np.load(

        RAW_LABEL_DIR / "test_label_idxs.npy"

    ).astype(np.int64)



    lda = joblib.load(

        MATCHED_DIR / "lda_model_11d.joblib"

    )



    X_train_lda_full = lda.transform(

        X_train_pca64_full

    )



    X_val_lda_full = lda.transform(

        X_val_pca64_full

    )



    X_test_lda_full = lda.transform(

        X_test_pca64_full

    )



    print("\nFull shared representation loaded:")

    print("  Train:", X_train_lda_full.shape)

    print("  Val  :", X_val_lda_full.shape)

    print("  Test :", X_test_lda_full.shape)



    return (

        X_train_lda_full,

        y_train_full,

        X_val_lda_full,

        y_val_full,

        X_test_lda_full,

        y_test_full,

    )





# ============================================================

# 5. CREATE ONE MATCHED SEED

# ============================================================



def create_seed_data(

    seed: int,

    results_dir: Path,

    run_name: str,

    full_data,

):



    (

        X_train_lda_full,

        y_train_full,

        X_val_lda_full,

        y_val_full,

        X_test_lda_full,

        y_test_full,

    ) = full_data



    # Preserve EXACT original protocol:

    # train uses seed

    # validation uses seed + 1

    # full held-out test set is only reordered using seed + 2



    train_idx = sample_exact_per_class(

        y_train_full,

        TRAIN_PER_CLASS,

        seed,

        "train",

    )



    val_idx = sample_exact_per_class(

        y_val_full,

        VAL_PER_CLASS,

        seed + 1,

        "validation",

    )



    test_idx = use_all_indices(

        y_test_full,

        seed + 2

    )



    X_train = X_train_lda_full[train_idx]

    y_train = y_train_full[train_idx]



    X_val = X_val_lda_full[val_idx]

    y_val = y_val_full[val_idx]



    X_test = X_test_lda_full[test_idx]

    y_test = y_test_full[test_idx]



    print("\n12-class fixed QK data:")

    print(

        f"  Train: X={X_train.shape}, "

        f"counts={np.bincount(y_train, minlength=N_CLASSES)}"

    )

    print(

        f"  Val  : X={X_val.shape}, "

        f"counts={np.bincount(y_val, minlength=N_CLASSES)}"

    )

    print(

        f"  Test : X={X_test.shape}, "

        f"counts={np.bincount(y_test, minlength=N_CLASSES)}"

    )



    np.save(

        results_dir / f"{run_name}_train_indices_full.npy",

        train_idx

    )



    np.save(

        results_dir / f"{run_name}_val_indices_full.npy",

        val_idx

    )



    np.save(

        results_dir / f"{run_name}_test_indices_full.npy",

        test_idx

    )



    np.save(

        results_dir / f"{run_name}_X_train_pca64_lda11.npy",

        X_train

    )



    np.save(

        results_dir / f"{run_name}_X_val_pca64_lda11.npy",

        X_val

    )



    np.save(

        results_dir / f"{run_name}_X_test_pca64_lda11.npy",

        X_test

    )



    np.save(

        results_dir / f"{run_name}_y_train_12class.npy",

        y_train

    )



    np.save(

        results_dir / f"{run_name}_y_val_12class.npy",

        y_val

    )



    np.save(

        results_dir / f"{run_name}_y_test_12class.npy",

        y_test

    )



    split_summary = {

        "seed": seed,

        "train_seed": seed,

        "validation_seed": seed + 1,

        "test_order_seed": seed + 2,

        "train_counts": np.bincount(

            y_train,

            minlength=N_CLASSES

        ).tolist(),

        "val_counts": np.bincount(

            y_val,

            minlength=N_CLASSES

        ).tolist(),

        "test_counts": np.bincount(

            y_test,

            minlength=N_CLASSES

        ).tolist(),

        "num_train": int(len(y_train)),

        "num_val": int(len(y_val)),

        "num_test": int(len(y_test)),

    }



    with open(

        results_dir / f"{run_name}_split_summary.json",

        "w"

    ) as f:

        json.dump(

            split_summary,

            f,

            indent=2

        )



    return (

        X_train,

        y_train,

        X_val,

        y_val,

        X_test,

        y_test,

    )





# ============================================================

# 6. TRAIN-ONLY MIN-MAX SCALING TO [0,1]

# ============================================================



def rescale_train_only_to_01(

    X_train,

    X_val,

    X_test,

    results_dir,

    run_name,

):



    train_min = X_train.min(axis=0)

    train_max = X_train.max(axis=0)



    denom = np.maximum(

        train_max - train_min,

        1e-12

    )



    X_train_01 = np.clip(

        (X_train - train_min) / denom,

        0.0,

        1.0

    )



    X_val_01 = np.clip(

        (X_val - train_min) / denom,

        0.0,

        1.0

    )



    X_test_01 = np.clip(

        (X_test - train_min) / denom,

        0.0,

        1.0

    )



    np.save(

        results_dir / f"{run_name}_rescale_train_min.npy",

        train_min

    )



    np.save(

        results_dir / f"{run_name}_rescale_train_max.npy",

        train_max

    )



    return (

        X_train_01,

        X_val_01,

        X_test_01,

    )





# ============================================================

# 7. EXACT ORIGINAL FIXED QUANTUM KERNEL

# ============================================================



def build_fixed_feature_map():



    from qiskit.circuit.library import ZZFeatureMap



    feature_map = ZZFeatureMap(

        feature_dimension=N_QUBITS,

        reps=FEATURE_MAP_REPS,

        entanglement=ENTANGLEMENT,

        parameter_prefix="x",

    )



    feature_params = sorted(

        list(feature_map.parameters),

        key=lambda p: p.name

    )



    print("\nFixed 12-class quantum kernel circuit:")

    print("  Circuit         : ZZFeatureMap(x)")

    print("  Feature params  :", len(feature_params))

    print("  Trainable params: 0")



    return feature_map, feature_params





def get_aer_backend():



    if not USE_AER:

        return None



    from qiskit_aer import AerSimulator



    backend = AerSimulator(

        method="statevector",

        device="GPU" if USE_AER_GPU else "CPU",

        precision=AER_PRECISION,

    )



    print(

        f"Using AerSimulator: "

        f"GPU={USE_AER_GPU}, "

        f"precision={AER_PRECISION}"

    )



    return backend





def make_bound_circuit(

    circuit,

    feature_params,

    x

):



    bind_dict = {

        p: float(v)

        for p, v in zip(

            feature_params,

            x

        )

    }



    return circuit.assign_parameters(

        bind_dict,

        inplace=False

    )





def compute_statevectors(

    X,

    circuit,

    feature_params,

    backend=None,

    batch_name=""

):



    if backend is not None:



        from qiskit import transpile



        circuits = []



        for idx, x in enumerate(X):



            bound = make_bound_circuit(

                circuit,

                feature_params,

                x

            )



            bound.save_statevector()



            bound.name = (

                f"{batch_name}_sv_{idx}"

            )



            circuits.append(bound)



        print(

            f"  Transpiling {len(circuits)} "

            f"circuits for {batch_name}..."

        )



        tcirc = transpile(

            circuits,

            backend

        )



        print(

            f"  Running Aer statevector simulation "

            f"for {batch_name}..."

        )



        res = backend.run(

            tcirc

        ).result()



        states = []



        for i in range(len(circuits)):



            states.append(

                np.asarray(

                    res.get_statevector(i),

                    dtype=np.complex64

                )

            )



        return np.vstack(states)



    from qiskit.quantum_info import Statevector



    states = []



    for x in X:



        bound = make_bound_circuit(

            circuit,

            feature_params,

            x

        )



        states.append(

            np.asarray(

                Statevector.from_instruction(

                    bound

                ).data,

                dtype=np.complex64

            )

        )



    return np.vstack(states)





def quantum_kernel_from_states(

    states_a,

    states_b

):



    return np.asarray(

        np.abs(

            states_a @ states_b.conj().T

        ) ** 2,

        dtype=np.float32

    )





# ============================================================

# 8. ORIGINAL VALIDATION-BASED C SELECTION

# ============================================================



def select_best_svc_c(

    K_train,

    y_train,

    K_val_train,

    y_val,

    results_dir,

    run_name,

) -> Dict[str, Any]:



    rows = []



    best = {

        "C": None,

        "val_accuracy": -1.0,

        "val_macro_f1": -1.0,

        "val_weighted_f1": -1.0,

    }



    for C in SVC_C_GRID:



        clf = SVC(

            kernel="precomputed",

            C=C,

            class_weight="balanced"

        )



        clf.fit(

            K_train,

            y_train

        )



        y_val_pred = clf.predict(

            K_val_train

        )



        val_acc = accuracy_score(

            y_val,

            y_val_pred

        )



        val_macro_f1 = f1_score(

            y_val,

            y_val_pred,

            average="macro",

            zero_division=0

        )



        val_weighted_f1 = f1_score(

            y_val,

            y_val_pred,

            average="weighted",

            zero_division=0

        )



        row = {

            "C": C,

            "val_accuracy": val_acc,

            "val_macro_f1": val_macro_f1,

            "val_weighted_f1": val_weighted_f1,

        }



        rows.append(row)



        if (

            val_macro_f1,

            val_acc

        ) > (

            best["val_macro_f1"],

            best["val_accuracy"]

        ):

            best = row



    pd.DataFrame(rows).to_csv(

        results_dir /

        f"{run_name}_svc_c_validation_results.csv",

        index=False

    )



    print("\nValidation C-selection:")

    print(

        pd.DataFrame(rows).to_string(

            index=False

        )

    )



    print(

        f"\nSelected C={best['C']}"

    )



    return best





# ============================================================

# 9. RUN ONE SEED

# ============================================================



def run_one_seed(

    seed: int,

    full_data,

    circuit,

    feature_params,

    backend,

):



    total_start = time.perf_counter()



    run_name = (

        f"fixed_qk_12class_50_seed{seed}"

    )



    results_dir = (

        BASE_RESULTS_DIR /

        f"seed_{seed}"

    )



    results_dir.mkdir(

        parents=True,

        exist_ok=True

    )



    print("\n")

    print("=" * 80)

    print(

        f"12C-QK-50/class | SEED {seed}"

    )

    print("=" * 80)



    save_system_info(

        seed,

        run_name,

        results_dir

    )



    (

        X_train,

        y_train,

        X_val,

        y_val,

        X_test,

        y_test,

    ) = create_seed_data(

        seed,

        results_dir,

        run_name,

        full_data,

    )



    (

        X_train_q,

        X_val_q,

        X_test_q,

    ) = rescale_train_only_to_01(

        X_train,

        X_val,

        X_test,

        results_dir,

        run_name,

    )



    kernel_start = time.perf_counter()



    print("\nComputing train statevectors...")



    sv_train = compute_statevectors(

        X_train_q,

        circuit,

        feature_params,

        backend,

        batch_name=f"train_seed{seed}",

    )



    print("\nComputing validation statevectors...")



    sv_val = compute_statevectors(

        X_val_q,

        circuit,

        feature_params,

        backend,

        batch_name=f"val_seed{seed}",

    )



    print("\nComputing test statevectors...")



    sv_test = compute_statevectors(

        X_test_q,

        circuit,

        feature_params,

        backend,

        batch_name=f"test_seed{seed}",

    )



    print("\nComputing quantum kernels...")



    K_train = quantum_kernel_from_states(

        sv_train,

        sv_train

    )



    K_train = (

        K_train +

        KERNEL_RIDGE *

        np.eye(

            len(y_train),

            dtype=np.float32

        )

    )



    K_val_train = quantum_kernel_from_states(

        sv_val,

        sv_train

    )



    K_test_train = quantum_kernel_from_states(

        sv_test,

        sv_train

    )



    kernel_time = (

        time.perf_counter() -

        kernel_start

    )



    # Save kernels/prediction-relevant artifacts.

    np.save(

        results_dir /

        f"{run_name}_kernel_train.npy",

        K_train

    )



    np.save(

        results_dir /

        f"{run_name}_kernel_val_train.npy",

        K_val_train

    )



    np.save(

        results_dir /

        f"{run_name}_kernel_test_train.npy",

        K_test_train

    )



    best = select_best_svc_c(

        K_train,

        y_train,

        K_val_train,

        y_val,

        results_dir,

        run_name,

    )



    best_C = best["C"]



    print("\nTraining final QSVC...")



    clf = SVC(

        kernel="precomputed",

        C=best_C,

        class_weight="balanced"

    )



    clf.fit(

        K_train,

        y_train

    )



    y_pred = clf.predict(

        K_test_train

    )



    acc = accuracy_score(

        y_test,

        y_pred

    )



    macro_f1 = f1_score(

        y_test,

        y_pred,

        average="macro",

        zero_division=0

    )



    weighted_f1 = f1_score(

        y_test,

        y_pred,

        average="weighted",

        zero_division=0

    )



    np.save(

        results_dir /

        f"{run_name}_predictions.npy",

        y_pred

    )



    report_text = classification_report(

        y_test,

        y_pred,

        labels=list(range(N_CLASSES)),

        target_names=[

            CLASS_NAMES[i]

            for i in range(N_CLASSES)

        ],

        zero_division=0,

    )



    report_dict = classification_report(

        y_test,

        y_pred,

        labels=list(range(N_CLASSES)),

        target_names=[

            CLASS_NAMES[i]

            for i in range(N_CLASSES)

        ],

        output_dict=True,

        zero_division=0,

    )



    with open(

        results_dir /

        f"{run_name}_classification_report.txt",

        "w"

    ) as f:

        f.write(report_text)



    pd.DataFrame(

        report_dict

    ).transpose().to_csv(

        results_dir /

        f"{run_name}_classification_report.csv"

    )



    cm = confusion_matrix(

        y_test,

        y_pred,

        labels=list(range(N_CLASSES))

    )



    pd.DataFrame(

        cm,

        index=[

            CLASS_NAMES[i]

            for i in range(N_CLASSES)

        ],

        columns=[

            CLASS_NAMES[i]

            for i in range(N_CLASSES)

        ],

    ).to_csv(

        results_dir /

        f"{run_name}_confusion_matrix.csv"

    )



    total_time = (

        time.perf_counter() -

        total_start

    )



    summary = {

        "seed": seed,

        "train_sampling_seed": seed,

        "validation_sampling_seed": seed + 1,

        "test_order_seed": seed + 2,

        "run_name": run_name,

        "task": "12-class fixed quantum kernel",

        "representation": "PCA64 -> LDA11",

        "encoding": "[0,1]",

        "train_per_class": TRAIN_PER_CLASS,

        "val_per_class": VAL_PER_CLASS,

        "test_protocol":

            "all available held-out test samples",

        "num_train": int(len(y_train)),

        "num_val": int(len(y_val)),

        "num_test": int(len(y_test)),

        "selected_C": float(best_C),

        "val_macro_f1":

            float(best["val_macro_f1"]),

        "val_accuracy":

            float(best["val_accuracy"]),

        "test_accuracy": float(acc),

        "test_macro_f1": float(macro_f1),

        "test_weighted_f1":

            float(weighted_f1),

        "kernel_time_sec":

            float(kernel_time),

        "total_wall_time_sec":

            float(total_time),

    }



    pd.DataFrame(

        [summary]

    ).to_csv(

        results_dir /

        f"{run_name}_results_summary.csv",

        index=False

    )



    with open(

        results_dir /

        f"{run_name}_results_summary.json",

        "w"

    ) as f:

        json.dump(

            summary,

            f,

            indent=2

        )



    print("\nRESULT")

    print(

        f"Seed {seed}: "

        f"accuracy={acc:.6f}, "

        f"macro-F1={macro_f1:.6f}, "

        f"weighted-F1={weighted_f1:.6f}, "

        f"C={best_C}"

    )



    return summary





# ============================================================

# 10. FIVE-SEED AGGREGATION

# ============================================================



def save_aggregate_results(

    results

):



    aggregate_dir = (

        BASE_RESULTS_DIR /

        "aggregate"

    )



    aggregate_dir.mkdir(

        parents=True,

        exist_ok=True

    )



    df = pd.DataFrame(results)



    df.to_csv(

        aggregate_dir /

        "fixed_qk_12class_5seed_all_results.csv",

        index=False

    )



    metric_columns = [

        "test_accuracy",

        "test_macro_f1",

        "test_weighted_f1",

        "kernel_time_sec",

        "total_wall_time_sec",

    ]



    summary_rows = []



    for metric in metric_columns:



        values = df[metric].to_numpy(

            dtype=float

        )



        summary_rows.append({

            "metric": metric,

            "n": len(values),

            "mean": np.mean(values),

            "std": np.std(

                values,

                ddof=1

            ),

            "min": np.min(values),

            "max": np.max(values),

        })



    summary_df = pd.DataFrame(

        summary_rows

    )



    summary_df.to_csv(

        aggregate_dir /

        "fixed_qk_12class_5seed_mean_std.csv",

        index=False

    )



    print("\n")

    print("=" * 80)

    print("FIVE-SEED INDIVIDUAL RESULTS")

    print("=" * 80)



    print(

        df[

            [

                "seed",

                "selected_C",

                "test_accuracy",

                "test_macro_f1",

                "test_weighted_f1",

            ]

        ].to_string(

            index=False

        )

    )



    print("\n")

    print("=" * 80)

    print("FIVE-SEED SUMMARY")

    print("=" * 80)



    print(

        summary_df.to_string(

            index=False

        )

    )



    macro_mean = df[

        "test_macro_f1"

    ].mean()



    macro_sd = df[

        "test_macro_f1"

    ].std(ddof=1)



    acc_mean = df[

        "test_accuracy"

    ].mean()



    acc_sd = df[

        "test_accuracy"

    ].std(ddof=1)



    weighted_mean = df[

        "test_weighted_f1"

    ].mean()



    weighted_sd = df[

        "test_weighted_f1"

    ].std(ddof=1)



    print("\nManuscript-ready:")

    print(

        f"Accuracy: {acc_mean:.4f} ± "

        f"{acc_sd:.4f}"

    )

    print(

        f"Macro-F1: {macro_mean:.4f} ± "

        f"{macro_sd:.4f}"

    )

    print(

        f"Weighted-F1: {weighted_mean:.4f} ± "

        f"{weighted_sd:.4f}"

    )





# ============================================================

# 11. MAIN

# ============================================================



def main():



    print("=" * 80)

    print(

        "Reviewer 1 Concern 4: "

        "Five-seed 12-class fixed-QK evaluation"

    )

    print("=" * 80)



    print("Seeds:", SEEDS)



    full_data = (

        load_full_pca64_lda11_data()

    )



    # Feature map/backend are identical for all seeds.

    circuit, feature_params = (

        build_fixed_feature_map()

    )



    backend = get_aer_backend()



    results = []



    for seed in SEEDS:



        summary = run_one_seed(

            seed,

            full_data,

            circuit,

            feature_params,

            backend,

        )



        results.append(summary)



        # Save progressive aggregate so completed

        # seeds are not lost if a later run fails.

        pd.DataFrame(

            results

        ).to_csv(

            BASE_RESULTS_DIR /

            "completed_seed_results.csv",

            index=False

        )



    save_aggregate_results(

        results

    )



    print("\nAll five seeds completed.")

    print(

        "Results:",

        BASE_RESULTS_DIR

    )





if __name__ == "__main__":

    main()
