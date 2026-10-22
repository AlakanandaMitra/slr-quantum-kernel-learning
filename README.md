
# Supervised Latent Restructuring and Adaptive Quantum Kernels



Reproducibility repository for the IEEE Access manuscript:



**Supervised Latent Restructuring and Adaptive Quantum Kernels for Small-Data Plant Disease Classification**



## Overview



This repository contains the code and reproducibility artifacts for experiments comparing classical machine-learning models, fixed quantum kernels, and gamma-only Quantum Kernel Alignment (QKA) for small-data apple-leaf disease classification.



The workflow uses EfficientNet-B0 image embeddings followed by PCA64 and supervised LDA11 latent restructuring. Experiments include fine-grained 12-class classification and a coarse three-group taxonomy.



The repository contains the exact sampling indices and lightweight numerical outputs used in the reported experiments. Large intermediate simulation artifacts and superseded exploratory experiments are intentionally excluded.




## Dataset



The experiments use the Plant Pathology 2021 - FGVC8 apple-leaf image dataset.



Raw dataset images are not redistributed in this repository. Users should obtain the dataset separately before regenerating EfficientNet-B0 embeddings.



The repository includes the fine-grained label arrays required by the published quantum-kernel experiments, but excludes the full 1280-dimensional EfficientNet-B0 embeddings because they can be regenerated from the source images.



## Repository structure



```text

data/

  embeddings/efficientnet_b0/

    Fine-grained label arrays



results/

  matched_representation/

    PCA64 arrays, fitted PCA64/LDA11 models, and LDA fitting indices

  published/

    Lightweight numerical results underlying the manuscript



splits/

  12class/

    Exact five-seed 12-class sampling indices

  three_group/

    Exact three-group sampling indices for 100, 150, and 210 samples/group



src/

  data/

    Dataset utilities

  embeddings/

    EfficientNet-B0 feature extraction

  classical/

    Classical baseline and matched classical experiments

  quantum/

    Fixed quantum-kernel and gamma-only QKA experiments

  experiments/

    Ablations, sensitivity analyses, target controls, and single-seed QKA experiments

  analysis/

    Aggregation, statistical analysis, diagnostics, and plotting

```



See `MANIFEST.md` for the detailed mapping between experiments, code, exact splits, and published numerical outputs.


## Experimental protocol

The principal pipeline is:

1. Extract 1280-dimensional EfficientNet-B0 image embeddings.
2. Construct the PCA64 representation.
3. Apply supervised LDA11 latent restructuring.
4. Evaluate classical baselines.
5. Evaluate the fixed quantum kernel.
6. Train gamma-only QKA.
7. Repeat matched evaluations across five seeds.
8. Evaluate dimensionality, representation fitting, ridge sensitivity, target controls, and kernel diagnostics.

### Five-seed protocol

The matched evaluations use seeds: `42, 52, 62, 72, 82`.

The exact sampling indices used for the reported experiments are included under `splits/`.

### Included representation artifacts

The repository includes:

- `train_embeddings_pca.npy`
- `valid_embeddings_pca.npy`
- `test_embeddings_pca.npy`
- `pca_model_64d.joblib`
- `lda_model_11d.joblib`
- `train_indices.npy`

These are stored under `results/matched_representation/`.

## Reproducing the experiments

Run all commands from the repository root.

### EfficientNet-B0 embedding extraction

```bash
python src/embeddings/extract_efficientnet_b0.py --help
```

### Five-seed 12-class fixed quantum kernel

```bash
python src/quantum/run_fixed_qk_12class_5seed.py
```

### Five-seed matched 12-class classical evaluation

```bash
python src/classical/run_matched_classical_12class_5seed.py
```

### Gamma-only QKA

```bash
python src/quantum/qka_gamma_only_soft_3group.py --help
```

### Single-seed soft-target γ-only QKA



The initial single-seed three-group experiments used γ-only QKA with

11 trainable feature-scaling parameters and no trainable post-encoding

ansatz. Experiments were evaluated at 100, 150, and 210 training

samples per group.



The `[0,1]` implementation is:



```bash

python src/experiments/single_seed_qka/qka_soft_target_3group.py

```



The `[0,π]` encoding-sensitivity implementation is:



```bash

python src/experiments/single_seed_qka/qka_soft_target_3group_0pi.py

```



Saved numerical summaries supporting Table 9 and Supplementary

Tables S8--S9 are provided in `results/published/single_seed_qka/`.


### Matched three-group baselines

```bash
python src/classical/run_matched_multiseed_baselines.py --help
```

### PCA11 representation ablation

```bash
python src/experiments/run_pca11_ablation.py --help
```

### PCA dimensionality sensitivity

```bash
python src/experiments/run_pca_dim_sensitivity.py --help
```

## Computing environment

The principal reported environment used:

- Python 3.10.20
- PyTorch 2.5.1
- Qiskit 1.3.2
- Qiskit Aer 0.15.1
- scikit-learn 1.7.2
- XGBoost 3.2.0

GPU-enabled quantum simulation was performed with Qiskit Aer on NVIDIA L40S resources.

See `environment.yml` and `requirements.txt` for dependency information.

## Published numerical results

Compact numerical outputs underlying the manuscript are provided under `results/published/`.

These include seed-level summaries, aggregate statistics, representation ablations, ridge sensitivity, target controls, kernel diagnostics, and alignment-performance analyses.

Large statevectors, full kernel matrices, trained classifier binaries, and full EfficientNet-B0 embeddings are intentionally excluded.

## Citation

If you use this code or the associated experimental results, please cite:

Mitra, A., Fleisher, D. H., Ray, C., and Reddy, V. R. "Supervised Latent Restructuring for Fixed and Adaptive Quantum Kernel Learning in Small-Data Image Classification." IEEE Access, accepted for publication, 2026.

The DOI and final bibliographic details will be added when available.

See [CITATION.cff](CITATION.cff) for citation metadata.

## License

See [LICENSE](LICENSE) for the applicable software license.

The original Plant Pathology 2021 dataset is not redistributed and remains subject to its original terms of use. Third-party dependencies retain their respective licenses.
