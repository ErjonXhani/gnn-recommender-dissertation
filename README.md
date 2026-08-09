# Graph Neural Networks for Recommendation Systems

## Enhancing User–Item Interaction Modelling for E-Commerce Applications

**MSc Artificial Intelligence Dissertation**

Author: Erjon Xhani  
Supervisor: Dr Oleksandr Galkin  
City St George's, University of London

---

## Overview

This project compares three graph neural network recommenders — LightGCN, NGCF and
GraphSAGE — against non-graph baselines on the Amazon Reviews 2023 Electronics dataset,
and investigates whether product metadata improves recommendation performance.

Two evaluation protocols are used throughout: leave-last-out, which is common in the
literature, and a global timestamp split designed to prevent temporal lookahead of the
kind described by Ji et al. (2023). The choice of evaluation protocol ultimately had a
larger effect on the conclusions than the choice of graph architecture.

## Key findings

- **The ranking of models depends strongly on the evaluation protocol.** Under
  leave-last-out, LightGCN improves on matrix factorisation by 27%. Under the global
  timestamp split, matrix factorisation beats LightGCN by a factor of 2.9. Popularity
  moves from fourth place to last.

- **Metadata helps LightGCN but not matrix factorisation**, under both protocols. This
  suggests that content and graph propagation are complementary rather than
  interchangeable.

- **On cold items the ranking reverses.** GraphSAGE, the weakest model overall, becomes
  the strongest on items unseen during training — the setting where its feature
  aggregation should be most useful.

- **A properly tuned baseline closes most of the gap that graph models are reported to open.** On Gowalla, a tuned
  BPR-MF beats the published matrix factorisation baseline by 20%, reducing NGCF's
  reported advantage from 11.7% to 1.1%.

- **LightGCN becomes increasingly expensive as the dataset grows.** Its wall-clock
  training cost relative to matrix factorisation widens from 14× to 22× between the
  small and medium subsets.

## Repository layout

```text
src/         reusable modules imported by the notebooks
notebooks/   analysis and experiments, in numbered order
results/     saved metrics, per-user scores and cached training histories
figures/     figures used in the dissertation
docs/        research log and report notes
data/        datasets (gitignored, see below)
```

## Notebooks

| Notebook | Purpose |
|---|---|
| `01_reviews_dataset_statistics` | Raw review data: scale, sparsity, rating and activity distributions |
| `02_metadata_dataset_exploration` | Product metadata: field completeness, coverage, feature selection |
| `03_data_preprocessing_earliest` | Filtering, iterative 5-core, deduplication, identifier encoding |
| `03_validation_against_official_benchmark_earliest` | Reconciliation against the official released benchmark files |
| `04_train_validation_test_split_earliest` | Both splitting protocols, with leakage and coverage checks |
| `05_nested_dataset_subsets` | Nested subsets of increasing size for the scaling study |
| `06_graph_construction` | Bipartite graphs, normalised adjacencies, structural verification |
| `07_item_features` | Text encoding with BLaIR, MiniLM and TF-IDF+SVD; leakage audit |
| `08_evaluation_and_baselines` | Evaluator verification; Popularity, ItemKNN and BPR-MF |
| `09_models_and_verification` | Model implementations; reproduction of LightGCN and NGCF on Gowalla |
| `10_hyperparameter_tuning` | Tuning and the collaborative comparison across both protocols |
| `11_metadata_and_cold_start` | Metadata ablations, encoder comparison, cold-start evaluation |
| `12_significance_scaling_synthesis` | Significance tests, scaling study, final tables and figures |

Two notebooks retain the earlier duplicate-resolution policy for comparison:
`03_data_preprocessing.ipynb` and `03_validation_against_official_benchmark.ipynb` keep
the **latest** duplicate review per user–product pair, while the `_earliest` versions keep
the **earliest** and are the ones used for every result in the dissertation.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

PyTorch needs to match your CUDA version; see the note at the top of
`requirements.txt`.

`requirements.txt` lists the direct dependencies pinned to the versions used for every
reported result. `requirements_full.txt` is a complete environment snapshot from
`pip freeze`, for exact reproduction.

## Data

The raw data is not in the repository. Download the Electronics category of
[Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) (Hou et al., 2024) and
place the review and metadata files under `data/raw/`. Notebook 03 produces the processed data used by the later notebooks.

The Gowalla files used for the reproductions download automatically in Notebook 09, from
the LightGCN-PyTorch repository.

## Reproducing the results

Training runs cache their histories and metrics to `results/`, keyed by model,
protocol, hyperparameters and seed. Re-running a notebook reads from that cache rather
than retraining, so all notebooks execute in minutes once the data has been preprocessed.

Deleting a cache directory forces the corresponding runs to be recomputed. The complete set of experiments takes roughly 70 GPU-hours from scratch. 

## Hardware

The experiments were run locally on an NVIDIA GeForce RTX 5070 Ti Laptop GPU with
12 GB VRAM and 32 GB system memory. Wall-clock training times reported in the project
are therefore specific to this hardware configuration.

## References

### Data and models used

- **Amazon Reviews 2023**, Electronics category — Hou et al. (2024),
  [amazon-reviews-2023.github.io](https://amazon-reviews-2023.github.io/), arXiv:2403.03952.
  The preprocessing here reproduces the authors' official 5-core benchmark, which is used
  to validate it in Notebook 03.
- **Gowalla train/test split** — downloaded in Notebook 09 from
  [gusye1234/LightGCN-PyTorch](https://github.com/gusye1234/LightGCN-PyTorch). The split
  originates with NGCF (Wang et al., 2019) and was reused by LightGCN.
- **BLaIR** (`hyp1231/blair-roberta-base`) — released by the dataset authors, pretrained on
  Amazon Reviews 2023 metadata.
- **MiniLM** (`sentence-transformers/all-MiniLM-L6-v2`) — general-purpose sentence encoder,
  used as a control.

### Implementations

All models are implemented from the published equations in `src/models.py`; no model
code is copied. The released LightGCN PyTorch implementation was consulted to match two
implementation details not specified clearly in the paper: user-uniform negative
sampling and the factor of one half in the regularisation term. Both are documented and
tested in Notebook 09.

### Papers

- He et al. (2020) *LightGCN*. SIGIR. arXiv:2002.02126
- Wang et al. (2019) *Neural Graph Collaborative Filtering*. SIGIR. arXiv:1905.08108
- Hamilton et al. (2017) *Inductive Representation Learning on Large Graphs*. NeurIPS
- Rendle et al. (2009) *BPR: Bayesian Personalized Ranking from Implicit Feedback*. UAI
- Ji et al. (2023) *A Critical Study on Data Leakage in Recommender System Offline
  Evaluation*. ACM TOIS. doi:10.1145/3569930
- Krichene and Rendle (2020) *On Sampled Metrics for Item Recommendation*. KDD
