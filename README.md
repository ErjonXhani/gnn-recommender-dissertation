
# Graph Neural Networks for Recommender Systems: Evaluating Performance and Robustness in E-Commerce Settings

**MSc Artificial Intelligence Dissertation**

**Author:** Erjon Xhani  
**Supervisor:** Oleksandr Galkin  
**Institution:** City St George's, University of London  
**Year:** 2026

## Overview

This repository contains the experimental implementation and analysis for an MSc dissertation investigating how conclusions about graph-based recommendation depend on evaluation protocol, model configuration, product metadata and cold-item evaluation.

Using the **Amazon Reviews 2023 Electronics** dataset, the project compares three graph-based recommenders (**LightGCN, NGCF and GraphSAGE**) with three non-graph methods (**Popularity, ItemKNN and BPR-MF**). The experiments cover:

- **Evaluation protocols:** Leave-last-out and global timestamp splits.
- **Graph propagation:** Depth, regularisation sensitivity and computational scaling.
- **Product metadata:** BLaIR features, with an exploratory comparison against MiniLM and TF-IDF with SVD.
- **Cold-item recommendation:** Ranking against the full catalogue and against cold items only.

Gowalla is used separately to check the LightGCN and NGCF implementations against published benchmark results. It is not a second dataset for the main experimental comparisons.

## Key findings

- **Evaluation design changes model rankings.** ItemKNN performed best under leave-last-out, whereas BPR-MF performed best under the global timestamp protocol. LightGCN was the strongest graph model under both.
- **Depth needs to be considered alongside regularisation.** Retuning regularisation at each propagation depth changed the interpretation for LightGCN and NGCF, while GraphSAGE benefited more consistently from additional layers.
- **Metadata effects depend on the model.** Adding BLaIR features improved all three graph models under both protocols, while BPR-MF's mean performance decreased.
- **Cold-item conclusions depend on the candidate set.** Performance was near zero when cold items competed with the full catalogue. With candidates restricted to cold items, GraphSAGE with BLaIR obtained the highest score.
- **Graph propagation has a growing computational cost.** LightGCN's measured per-epoch training time increased from 8.2 times to 16.0 times that of BPR-MF between the smallest and full dataset subsets. LightGCN training was not completed at full scale.

These findings apply to the experimental settings studied. The two evaluation protocols differ in more than the splitting rule alone. Full results, qualifications and limitations are discussed in the dissertation.

## Repository structure

```text
figures/      Figures produced for the dissertation
notebooks/    Numbered notebooks for experiments and analysis
results/      Saved metrics, per-user scores and experiment caches
src/          Reusable Python modules shared by the notebooks
data/         Local datasets (gitignored; not included in the repository)
```

The main workflow consists of **13 Jupyter notebooks**, supported by **8 reusable Python modules**. Each notebook documents the experimental steps, includes relevant outputs and ends with a summary of the main findings.

## Notebook guide

| Notebook | Purpose |
| --- | --- |
| `01_reviews_dataset_statistics.ipynb` | Explore review counts, activity patterns and sparsity. |
| `02_metadata_dataset_exploration.ipynb` | Examine product metadata coverage and available fields. |
| `03_01_data_preprocessing_earliest.ipynb` | Clean interactions, deduplicate using the earliest review, apply iterative five-core filtering and encode IDs. |
| `03_02_validation_against_official_benchmark_earliest.ipynb` | Check the final preprocessing against the official Electronics benchmark. |
| `04_train_validation_test_split_earliest.ipynb` | Build and validate both evaluation protocols, including temporal checks. |
| `05_nested_dataset_subsets.ipynb` | Construct nested dataset subsets for the scaling analysis. |
| `06_graph_construction.ipynb` | Build and verify the bipartite interaction graphs. |
| `07_item_features.ipynb` | Audit metadata and construct BLaIR, MiniLM and TF-IDF/SVD representations. |
| `08_evaluation_and_baselines.ipynb` | Verify ranking evaluation and implement non-graph baselines. |
| `09_models_and_verification.ipynb` | Implement and verify graph models; reproduce selected Gowalla benchmark results. |
| `10_hyperparameter_tuning.ipynb` | Tune models and conduct the interaction-only comparisons. |
| `11_metadata_and_cold_start.ipynb` | Run metadata comparisons, text-representation exploration and cold-item evaluation. |
| `12_significance_scaling_synthesis.ipynb` | Perform statistical tests, scaling analysis and results synthesis. |

**Preprocessing versions:** Two earlier notebooks, `03_data_preprocessing.ipynb` and `03_validation_against_official_benchmark.ipynb`, are retained for comparison. They use the *latest* review for duplicate user-product pairs. The final pipeline uses the `_earliest` notebooks, and **all results in the dissertation use the earliest-interaction policy**.

## Environment setup

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows (Command Prompt):

```bat
.venv\Scripts\activate
```

Or on macOS/Linux:

```bash
source .venv/bin/activate
```

Then install the project dependencies:

```bash
python -m pip install -r requirements.txt
```

`requirements.txt` lists the pinned direct dependencies. `requirements_full.txt` records the original environment from `pip freeze`; it is provided as an additional reference, although exact environment replication may depend on the operating system and GPU setup.

**PyTorch must be installed with a build compatible with the available CUDA environment**; see the note in `requirements.txt`.

## Data preparation

Raw data is **not included** in the repository. Download the Electronics review interactions and product metadata from the [Amazon Reviews 2023 dataset website](https://amazon-reviews-2023.github.io/) and place the files in `data/raw/` using the filenames expected by the notebooks.

Run the numbered notebooks in the order shown above. The final preprocessing retains the earliest user-product interaction and reapplies iterative five-core filtering after deduplication. It produces **15,473,536 interactions**, **1,641,026 users** and **368,228 products**. Most main experiments use a smaller, nested primary subset with **2,346,776 interactions**.

The final preprocessing and evaluation splits are checked against the official Electronics benchmark files in Notebooks 03 and 04.

**Gowalla:** Notebook 09 retrieves the released Gowalla files from the [LightGCN-PyTorch repository](https://github.com/gusye1234/LightGCN-PyTorch) for the external implementation checks.

## Reproducing and inspecting the experiments

Each notebook documents its inputs, checks and outputs. After preparing the dataset and environment, the notebooks can be run sequentially or their saved outputs can be inspected without rerunning the full pipeline.

Experiment histories, summary metrics and per-user scores are cached in `results/` by experimental configuration. **When the corresponding cached files are available**, later runs can reuse them instead of retraining. Missing or deleted caches require the relevant experiments to be executed again; the complete pipeline is computationally expensive.

The experiments were performed locally using an **NVIDIA GeForce RTX 5070 Ti Laptop GPU (12 GB VRAM)** and **32 GB system RAM**. Across the project, training and evaluation required approximately **60 hours of cumulative experiment runtime**, excluding development and code-writing time. This describes the original execution, not a guaranteed runtime on another machine.

The scaling study uses short training probes to measure per-epoch cost. LightGCN was **not trained to completion** on the full dataset subset because of its projected runtime.

## Implementation notes and external resources

The recommendation models were implemented for this project from their published formulations. The released LightGCN implementation was consulted when checking training details, including user-uniform negative sampling and regularisation.

The Gowalla reproduction is an **implementation check**, not a claim that Gowalla results are directly comparable with the main Electronics experiments.

- **Dataset and BLaIR:** [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/); Hou et al. (2026), *Bridging Language and Items for Retrieval and Recommendation: Benchmarking LLMs as Semantic Encoders*.
- **LightGCN:** He et al. (2020), *LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation*.
- **NGCF:** Wang et al. (2019), *Neural Graph Collaborative Filtering*.
- **GraphSAGE:** Hamilton, Ying and Leskovec (2017), *Inductive Representation Learning on Large Graphs*.
- **BPR:** Rendle et al. (2009), *BPR: Bayesian Personalized Ranking from Implicit Feedback*.

The feature experiments also use `hyp1231/blair-roberta-base` and `sentence-transformers/all-MiniLM-L6-v2`.

The dissertation contains the full bibliography and detailed discussion of the supporting literature.

## Generative AI disclosure

Generative AI was used as a supporting tool for literature exploration, implementation discussion and debugging, and report polishing. Each notebook includes a declaration of generative AI use at the end, while **Chapter 5 and Appendix C of the dissertation** provide the overall disclosure and reflection.

All reported numerical results were produced by the implemented experimental pipeline.

