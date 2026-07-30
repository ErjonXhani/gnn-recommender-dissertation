# Dissertation Report Notes

## Dataset

- Amazon Reviews 2023 Electronics dataset.
- Two datasets used:
  - Review interactions.
  - Product metadata.
- Linked using `parent_asin`.
- Metadata available for all products retained after preprocessing.

---

## Dataset Exploration

- Implemented streaming processing to avoid loading the full dataset into memory.
- Cached computed statistics to avoid repeated scans.
- Generated descriptive statistics and exploratory visualisations.
- Explored metadata completeness and attribute distributions.
- Verified metadata coverage between reviews and products.

---

## Metadata Selection

Reasons for selected features:
- High completeness.
- Relevant for item representation.

Selected:
- title
- store
- main_category
- categories
- features
- description
- details

Excluded:
- videos (high missingness)
- bought_together (high missingness)
- price (only 32.7% coverage)

Mention creation of combined metadata text.

---

## Data Preprocessing

Preprocessing pipeline:

- Stream processing of raw reviews.
- Field selection.
- Activity threshold analysis.
- Five-core filtering.
- Duplicate interaction analysis.
- Duplicate removal.
- Second five-core filtering.
- Metadata extraction.
- Metadata cleaning.
- Identifier encoding.
- Export processed datasets.

---

## Design Choices

Mention why:

- `parent_asin` selected.
- Five interactions chosen.
- Iterative five-core filtering required.
- One interaction retained per user-product pair.
- DuckDB used for large-scale duplicate analysis.
- Integer encoding required for GNN libraries.

---

## Dataset Statistics

Initial five-core:

- 15,873,192 interactions
- 1,676,376 users
- 374,249 products

Final dataset:

- 15,473,536 interactions
- 1,641,026 users
- 368,228 products

After resolving duplicate user–product interactions, the iterative 5-core filtering procedure was executed a second time. Removing duplicate interactions reduced the number of unique interactions for some users and products, causing a subset of entities to fall below the minimum interaction threshold. Reapplying the 5-core filtering ensured that every retained user and product satisfied the minimum of five interactions in the final dataset.

Mention:
- No duplicate user-product pairs.
- Every user/item has at least five interactions.
- 100% metadata coverage after preprocessing.

---

## Validation Against Official Benchmark

Mention:

- Independently implemented preprocessing from raw data.
- Compared against official Electronics 5-core benchmark.
- Exact match for:
  - users
  - products
  - interactions
  - user-item pairs
- 99.83% agreement in timestamps/ratings.
- Remaining 0.17% due only to duplicate-review policy.
- Verified discrepancy using original raw review data.

## Duplicate-Review Policy Validation (EARLIEST DUPLICATE KEPT)

- Investigated differences between the preprocessing pipeline and the official Amazon Reviews 2023 benchmark.
- Updated the duplicate-resolution strategy to retain the **earliest** review for duplicated user–product pairs.
- Reprocessed the dataset and validated it against the official benchmark.
- Validation confirmed identical dataset statistics and graph structure, with only **one** remaining interaction differing because of a duplicate-review tie case with identical timestamps.
- The revised preprocessing pipeline therefore reproduces the official benchmark almost exactly.

---

## Discussion

Mention:

- Independent reproduction increases confidence in preprocessing.
- Duplicate-review policy identified as only methodological difference.



---

## Train / Validation / Test Splitting

Implemented and evaluated two benchmark splitting protocols:

- Leave-last-out.
- Global absolute timestamp.

Mention:

- Both protocols were implemented independently from the official benchmark.
- Splitting logic moved into a reusable `src/splitting.py` module.
- Deterministic `(user_idx, timestamp, item_idx)` ordering used to resolve timestamp ties reproducibly.
- Original string identifiers used for benchmark validation because integer encodings are project-specific.
- Generated reusable train, validation, and test partitions for all subsequent experiments.

---

## Split Validation

Mention:

- Compared generated splits against the official Amazon Reviews 2023 benchmark.
- Global timestamp split reproduced the benchmark exactly.
- Leave-last-out differed for only 25 users (0.0015%).
- All remaining differences explained by timestamp ties.
- All automated integrity checks passed successfully.

---

## Evaluation Protocol Comparison

Mention:

- Leave-last-out evaluates every user but introduces temporal lookahead.
- Global timestamp preserves chronological ordering and completely removes lookahead.
- Quantified lookahead under leave-last-out to demonstrate its magnitude.
- Discuss how the choice of evaluation protocol may influence reported recommendation performance.

---

## Timestamp Tie Handling

Mention:

- Timestamp ties occur when multiple interactions share identical timestamps.
- Deterministic tie-breaking ensures reproducible dataset generation.
- Only 44 users (0.0027%) were affected by timestamp ties.
- Tie cases fully explain the remaining benchmark differences.

---

## Evaluation User Sampling

Mention:

- Evaluation performed on a fixed random sample of 50,000 users per protocol.
- Users sampled once using a fixed random seed.
- Same evaluation users reused across every model.
- Users are sampled, but items are never sampled during Top-K evaluation.
- Users without training history excluded from evaluation.

---

## Cold-Start Evaluation

Mention:

- Generated lists of users without training history.
- Generated lists of strictly cold items.
- Timestamp protocol produces substantially more cold items than leave-last-out.
- Timestamp split therefore supports a dedicated item cold-start experiment.
- Cold-item lists saved for reuse in later notebooks.

---

## Discussion

Mention:

- Comparison of leave-last-out versus global timestamp evaluation.
- Advantages and limitations of each protocol.
- Impact of temporal leakage on recommendation evaluation.
- Motivation for implementing and evaluating both splitting strategies.
- Justification for using the timestamp split for dedicated cold-start experiments.
- Importance of validating independently generated splits against the official benchmark to ensure reproducibility.

## Notebook 05 Scalability Dataset Construction

Mention:

- Three nested datasets (`small`, `medium`, `full`) created for the scalability experiments.
- Users sampled as nested prefixes of a single random permutation.
- Iterative 5-core filtering applied after sampling.
- User and item indices re-encoded while preserving mappings to the original global IDs.

---

## Dataset Calibration

Mention:

- Sampling fractions determined through a calibration experiment.
- Final subset sizes selected based on the resulting interaction counts after 5-core filtering.
- `small` dataset chosen to be comparable in scale to Amazon-Book.

---

## Dataset Characteristics

Mention:

- Final interaction counts for the three subsets.
- Mean user degree remained nearly constant across subset sizes.
- Mean item degree decreased as dataset size decreased.
- Subsets are not uniformly scaled versions of the original graph.

---

## Dataset Verification

Mention:

- Verified that the subsets are exactly nested.
- Confirmed that the `full` subset reproduced the Notebook 04 split counts.
- All integrity checks passed successfully.

---

## Experimental Setup

Mention:

- Generated both Leave-Last-Out and Timestamp splits for every subset.
- Original timestamp cutoffs reused across all subset levels.
- Timestamp protocol produced sufficient cold items for the planned cold-start evaluation.
- `small` selected as the primary experimental dataset.
- `medium` and `full` reserved for the scalability experiments.


# Notebook 06 - Graph Construction

Mention:

- Converted the training interactions into a bipartite user–item graph for GNN training.
- Assigned users and items to a shared node index space to construct a single adjacency matrix.
- Represented each interaction as bidirectional edges.
- Applied the standard LightGCN symmetric normalization (`D^-1/2 A D^-1/2`) without self-loops.
- Graphs were generated separately for each subset and splitting protocol.

---

## Graph Verification

Mention:

- Verified that no validation or test interactions appeared in the training graphs.
- Confirmed that all adjacency matrices were symmetric.
- Verified the normalization using the spectral radius (expected value = 1).
- Checked graph connectivity through connected component analysis.
- Validated the graph implementation against the Notebook 05 preprocessing outputs.

---

## Graph Statistics

Mention:

- Number of nodes and edges for each experimental dataset.
- Leave-last-out graphs consisted of one giant connected component plus isolated nodes.
- Timestamp graphs remained approximately 96% connected, with most disconnected nodes corresponding to cold users and items.
- Graph sizes were suitable for local GPU training.

---

## Implementation Choices

Mention:

- Graphs were rebuilt directly from the training interactions rather than stored on disk.
- Power iteration was used to estimate the spectral radius until convergence.
- GPU acceleration substantially reduced graph verification time.

---

## Experimental Pipeline

Mention:

- Graph construction uses only the training interactions.
- The same graph-building procedure is applied consistently across all dataset sizes and splitting protocols.
- The resulting graphs serve as the input for all graph-based recommendation models.



# Notebook 07 - Item Feature Construction

## Overview

Mention:

- Converted the cleaned product metadata into numeric item features for the recommendation models.
- Generated three text representations:
  - **BLaIR** (primary encoder)
  - **MiniLM** (general-purpose transformer baseline)
  - **TF-IDF + SVD** (classical baseline)
- Computed features once for all items and sliced them into each evaluation subset using the item mappings.

---

## Temporal Leakage in Metadata

Mention:

- Metadata was collected in 2023, after the interaction period used for evaluation.
- `Best Sellers Rank` is a popularity measure recorded after the prediction period and would introduce temporal leakage.
- It appeared in **56.3%** of items and was removed before encoding, with zero residual occurrences.
- `Date First Available` was retained because a product's release date is available at prediction time.
- Earlier preprocessing focused on data completeness and quality; this stage evaluated whether metadata fields were available at prediction time.

---

## Encoder Selection

Mention:

- Selected **BLaIR** as the primary encoder because it was pre-trained by the dataset authors on Amazon Reviews 2023 item metadata.
- Used **MiniLM** and **TF-IDF + SVD** as comparison baselines rather than equal alternatives.
- MiniLM evaluates whether a general-purpose language model performs competitively against a domain-specific encoder.
- TF-IDF + SVD provides a classical non-neural baseline.
- The comparison baselines are used only in the cold-start experiments, where item features are the only available information.
- The BLaIR paper reports that general embedding benchmarks do not necessarily predict recommendation performance, motivating its use as the primary encoder.

---

## Text Encoding

Mention:

- Followed the BLaIR model card by using CLS pooling with L2 normalisation.
- Measured the token-length distribution directly instead of estimating it from word counts.
- Average tokenisation ratio: **1.57 tokens per word**.
- Median sequence length: **402 tokens**.
- 95th percentile: **982 tokens**.
- Used **512 tokens** for BLaIR (RoBERTa's maximum), with **67.5%** of items fitting without truncation.
- Metadata order (title, store, category, features, description, details) means truncation primarily removes detailed specifications rather than core product descriptions.
- Used MiniLM with its native **256-token** input limit.
- Encoding times:
  - BLaIR: **19 minutes**
  - MiniLM: **7.3 minutes**
  - TF-IDF + SVD: **1.3 minutes**

---

## Embedding Quality

Mention:

- Used nearest-neighbour inspection to verify that embeddings captured meaningful semantic relationships.
- Mean cosine similarity between random item pairs decreased from **0.645** to **0.003** after centring.
- **165 of 768** embedding dimensions explain **95%** of the variance, indicating the embeddings remain well distributed rather than collapsing.
- TF-IDF + SVD retained **26.4%** of the original TF-IDF variance using 256 components.

---

## Categorical Features

Mention:

- One-hot encoded `main_category` (39 levels, including 6,632 items with no recorded category).
- Represented `store` only through the text embeddings rather than as a separate categorical feature.
- `store` contains **49,344** distinct values, with the 200 most frequent covering only **29.8%** of items, making explicit encoding impractical.
- Both `store` and `main_category` are already included in the textual metadata provided to the encoders.

---

## Feature Assembly

Mention:

- Centred and re-normalised the text embeddings before concatenating them with the categorical features.
- One-hot vectors already have unit length, allowing both feature blocks to contribute on comparable scales without manually tuning weights.
- Computed normalisation statistics over all items because product descriptions are available when items are listed and therefore do not introduce temporal leakage.
- Final feature dimensions:
  - **BLaIR:** 807
  - **MiniLM:** 423
  - **TF-IDF + SVD:** 295

---

## Feature Verification

Mention:

- Confirmed that all post-cutoff metadata fields were removed before encoding.
- Verified that all text embeddings were correctly centred and normalised.
- Confirmed exactly one category assignment per item.
- Verified alignment between the global feature matrices and every subset's local item mapping.
- Confirmed that every cold item has a valid feature vector.
- Verified that the feature loader reconstructs every subset/encoder combination correctly.

---

## Implementation

Mention:

- Implemented reusable feature preparation and loading utilities in `src/features.py`.
- Generated one global feature matrix per encoder and reconstructed subset feature matrices on demand instead of storing duplicate copies.

---

## Discussion

Mention:

- Metadata recorded after the interaction period represents a distinct form of temporal leakage that must be addressed before feature generation.
- Data completeness and temporal validity are separate criteria when selecting metadata features.
- Domain-specific language models may capture recommendation-relevant semantics more effectively than general-purpose encoders.
- Sequence length introduces a trade-off between computational cost and information retention.
- Item features are especially important in the cold-start setting because they provide the only available information for unseen items.
- Qualitative nearest-neighbour examples suggested that BLaIR emphasises product function more than MiniLM or TF-IDF + SVD, but this observation must be validated quantitatively through the cold-start experiments.



# Notebook 08 - Evaluation and Baseline Models

### Evaluation Methodology

- Explain why the evaluation framework was implemented and verified before any recommendation model.
- Describe the full-catalogue evaluation protocol and justify sampling users but never items (Krichene & Rendle, 2020).
- Explain the two evaluation protocols (leave-last-out and global timestamp) and why both are reported.
- Justify masking both training and validation interactions during test evaluation.
- Explain why HitRate@K and NDCG@K are reported for leave-last-out, while Recall@K is additionally reported for the timestamp split.

### Baseline Models

- Introduce Popularity, ItemKNN and BPR-MF as progressively stronger interaction-only baselines.
- Explain why BPR-MF is equivalent to 0-layer LightGCN and therefore provides the controlled comparison for graph propagation.
- Describe ItemKNN with shrunk cosine similarity and why shrinkage is required on sparse data.
- Explain BPR optimisation, uniform negative sampling, explicit L2 regularisation and early stopping.

### Hyperparameter Selection

- State that all hyperparameters were selected using validation data only.
- Mention the 10,000-user validation subset used during tuning and the full evaluation set used for early stopping and final reporting.
- Record the final hyperparameters selected for ItemKNN and BPR-MF under both protocols.

### Results

- Report that the evaluator passed all verification checks:
  - hand-computed synthetic example,
  - random recommender expectation,
  - independent NumPy implementation.
- Present baseline performance under both evaluation protocols.
- Highlight that the ranking of models changes completely between the two protocols.
- Report the reduction in Popularity performance under the leakage-free timestamp split.
- Report the improvement obtained by ItemKNN after adding shrinkage.
- Report the improvement obtained by tuning BPR-MF compared with literature defaults.

### Discussion

- Discuss why the evaluation protocol changes model ranking (Ji et al., 2023).
- Explain why Popularity benefits from temporal leakage under leave-last-out.
- Explain why cosine similarity overestimates similarities between very rare items and how shrinkage corrects this.
- Discuss why sparse interaction data limits matrix factorisation performance.
- Motivate the next chapter by explaining that graph propagation is intended to overcome sparse user histories.

### Key Quantitative Values

- 50,000 evaluation users
- 10,000 validation users
- 105,298 catalogue items
- 7.08 training interactions per user
- Popularity NDCG@10: **0.00752 → 0.00124**
- ItemKNN best shrinkage:
  - 200 (leave-last-out)
  - 500 (timestamp)
- BPR-MF best embedding dimension: **32**
- BPR-MF regularisation:
  - 0.05 (leave-last-out)
  - 0.08 (timestamp)
- Full-catalogue evaluation (no item sampling)