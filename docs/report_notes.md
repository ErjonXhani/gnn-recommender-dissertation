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
- Awaiting supervisor decision on whether to align with official benchmark.


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