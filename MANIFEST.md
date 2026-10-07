
# Reproducibility Manifest



This repository accompanies the IEEE Access manuscript on supervised latent

restructuring and quantum-kernel learning for small-data plant-disease

classification.



The repository contains publication-oriented code, exact sampling indices,

the fitted PCA64/LDA11 representation artifacts used by the reported

experiments, and lightweight numerical result summaries.



Large raw image data, full EfficientNet-B0 embeddings, quantum statevectors,

kernel matrices, and trained classifier binaries are intentionally excluded.



## Core representation



| Component | Repository location |

|---|---|

| PCA64 train representation | `results/matched_representation/train_embeddings_pca.npy` |

| PCA64 validation representation | `results/matched_representation/valid_embeddings_pca.npy` |

| PCA64 test representation | `results/matched_representation/test_embeddings_pca.npy` |

| PCA64 model | `results/matched_representation/pca_model_64d.joblib` |

| LDA11 model | `results/matched_representation/lda_model_11d.joblib` |

| LDA fitting indices | `results/matched_representation/train_indices.npy` |

| Fine-grained train labels | `data/embeddings/efficientnet_b0/train_label_idxs.npy` |

| Fine-grained validation labels | `data/embeddings/efficientnet_b0/valid_label_idxs.npy` |

| Fine-grained test labels | `data/embeddings/efficientnet_b0/test_label_idxs.npy` |




## Main experiments



### Five-seed 12-class classical evaluation



- Code: `src/classical/run_matched_classical_12class_5seed.py`

- Single-run implementation: `src/classical/run_matched_classical_12class_50.py`

- Exact splits: `splits/12class/`

- Results: `results/published/12class/classical/`



### Five-seed 12-class fixed quantum kernel



- Code: `src/quantum/run_fixed_qk_12class_5seed.py`

- Exact splits: `splits/12class/`

- Results: `results/published/12class/fixed_qk/`



### Five-seed three-group RBF-SVM and fixed quantum kernel



- Code: `src/classical/run_matched_multiseed_baselines.py`

- Exact splits: `splits/three_group/`

- Results: `results/published/three_group/baselines/`



### Five-seed gamma-only QKA



- Code: `src/quantum/qka_gamma_only_soft_3group.py`

- Exact splits: `splits/three_group/`

- Results: `results/published/three_group/qka/`

- Aggregate statistics: `results/published/statistics/`




## Controls and ablations



### PCA11 dimensionality-matched control



- Code: `src/experiments/run_pca11_ablation.py`

- Results: `results/published/pca11/`



### PCA dimensionality sensitivity



- Code: `src/experiments/run_pca_dim_sensitivity.py`

- Results: `results/published/pca_dimension/`



### Representation-fitting and direct-LDA audit



- Code: `src/experiments/audit_lda_classical_only.py`

- Additional audit: `src/experiments/audit_lda_classical_only_v2.py`

- Results: `results/published/ablations/lda_classical_only_summary.csv`



### Ridge sensitivity



- Code: `src/experiments/ridge_sensitivity_fixed_checkpoint.py`

- Results: `results/published/ablations/ridge_sensitivity_summary.csv`



### Silhouette stability



- Code: `src/analysis/bootstrap_silhouette_stability.py`

- Results: `results/published/ablations/silhouette_bootstrap_summary.csv`



### QKA target controls



- Code: `src/experiments/qka_gamma_only_target_control_3group.py`

- Aggregation: `src/analysis/aggregate_target_controls.py`

- Results: `results/published/target_controls/`



### Kernel diagnostics



- Code: `src/analysis/analyze_kernel_diagnostics.py`

- Results: `results/published/kernel_diagnostics/`



### Alignment-performance analysis



- Code: `src/analysis/analyze_alignment_vs_performance.py`

- Results: `results/published/alignment/`


## Excluded artifacts

The following are intentionally not stored in Git:

- Plant Pathology 2021 raw images
- Full 1280-dimensional EfficientNet-B0 embeddings
- Quantum statevectors
- Full kernel matrices
- Large trained model binaries
- Exploratory notebooks
- Superseded exploratory experiments not reported in the manuscript
- Debugging outputs and logs

The EfficientNet-B0 embeddings can be regenerated using `src/embeddings/extract_efficientnet_b0.py`.

## Single-seed three-group quantum-kernel experiments

### Soft-target γ-only QKA and encoding-range sensitivity

- `[0,1]` implementation: `src/experiments/single_seed_qka/qka_soft_target_3group.py`
- `[0,pi]` implementation: `src/experiments/single_seed_qka/qka_soft_target_3group_0pi.py`
- Published summaries: `results/published/single_seed_qka/`
- Training sizes: 100, 150, and 210 samples per group
- These experiments support the single-seed QKA results in Table 9, the QKA encoding-range comparison in Table S8, and the QKA runtime diagnostics in Table S9.
- The reported adaptive experiments optimize only the 11 feature-scaling parameters γ; no trainable post-encoding ansatz is used.
