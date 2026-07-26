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