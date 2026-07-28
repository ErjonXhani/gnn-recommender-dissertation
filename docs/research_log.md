## 23 July 2026 - Review and Metadata Dataset Statistics

### Completed
- Implemented streaming data processing for the Amazon Electronics review dataset.
- Added automatic caching of computed review dataset statistics.
- Validated the cache loading mechanism.
- Generated descriptive statistics and exploratory visualisations for the review dataset.
- Explored the Amazon Electronics metadata dataset.
- Analysed metadata completeness, duplicate product identifiers, store distribution, main category distribution, and price statistics.
- Verified metadata coverage between the review and metadata datasets.
- Selected candidate metadata features for the recommendation system based on completeness and relevance.

### Design Decisions
- Selected `parent_asin` as the product identifier because it consistently links review interactions with product metadata.
- Used streaming processing with `Counter` objects instead of loading the full datasets into memory to minimise RAM usage.
- Cached computed statistics and metadata exploration results to avoid repeated full-dataset scans.
- Excluded `videos` and `bought_together` from further processing due to their high proportion of missing values.
- Chose `title`, `store`, `main_category`, `categories`, `features`, `description`, and `details` as candidate metadata attributes for later preprocessing and graph construction.

### Key Findings
- The review dataset contains metadata for 100% of reviewed products and review interactions.
- The metadata dataset contains over 1.6 million products.
- Product price information is available for only 32.7% of products, making it unsuitable as a primary metadata feature.
- Product title, store, category, and technical metadata have high coverage and are suitable candidates for enriching item representations.

### Next Steps
- Implement the data preprocessing pipeline.
- Clean and standardise review and metadata fields.
- Handle missing metadata values.
- Filter inactive users and products.
- Encode user and product identifiers.
- Merge review interactions with selected metadata attributes.
- Prepare temporal train/validation/test splits.

## 24 July 2026 - Data Preprocessing

### Completed
- Started and completed the main data preprocessing pipeline for the Amazon Electronics dataset.
- Evaluated minimum user and product activity thresholds and selected a threshold of five interactions.
- Streamed and validated the raw review dataset without loading it fully into memory.
- Retained valid interaction fields: `user_id`, `parent_asin`, `rating`, `timestamp`, and `verified_purchase`.
- Applied iterative five-core filtering until every retained user and product had at least five interactions.
- Analysed duplicate user–product interactions using DuckDB.
- Removed exact duplicate records and retained the most recent interaction for non-identical repeated user–product pairs.
- Reapplied iterative five-core filtering after deduplication.
- Produced a final clean interaction dataset containing 15,473,536 unique interactions, 1,641,026 users, and 368,228 products.
- Extracted metadata only for products present in the final interaction dataset.
- Verified complete metadata coverage, with one metadata record for every retained product.
- Cleaned and standardised the selected metadata fields.
- Created text representations for categories, features, descriptions, and product details.
- Created a combined metadata text field for later item-feature generation.
- Encoded user and product identifiers as contiguous integer indices.
- Saved the encoded interaction dataset, encoded metadata dataset, user mapping, item mapping, and preprocessing summary files.

### Design Decisions
- Selected a minimum threshold of five interactions for both users and products to reduce graph sparsity while preserving a large amount of data.
- Used iterative five-core filtering because removing low-activity users and products can cause connected entities to fall below the threshold.
- Kept only one interaction per user–product pair to prevent duplicate pairs from appearing across temporal train, validation, and test sets.
- Retained the most recent interaction when a user had multiple non-identical interactions with the same product.
- Used DuckDB and partitioned Parquet files to process the large dataset efficiently.
- Retained `title`, `store`, `main_category`, `categories`, `features`, `description`, and `details` as product metadata.
- Replaced missing categorical metadata with explicit placeholders and converted complex metadata fields into clean text.
- Converted original string identifiers into contiguous integer indices for compatibility with graph-learning libraries.

### Key Findings
- The initial five-core dataset contained 15,873,192 interactions, 1,676,376 users, and 374,249 products.
- Duplicate analysis identified 219,618 repeated user–product pairs and 271,290 additional duplicate rows.
- After duplicate resolution and final five-core cleanup, the dataset contained:
  - 15,473,536 unique interactions
  - 1,641,026 users
  - 368,228 products
- Every retained user and product has at least five interactions.
- No duplicate user–product pairs remain.
- Metadata coverage is 100%, with exactly 368,228 metadata records for 368,228 retained products.
- No missing titles, stores, main categories, or combined metadata text remain after cleaning.

### Next Steps
- Create temporal training, validation, and test splits.
- Verify that each user has sufficient interactions in the training set.
- Prevent future interactions from leaking into earlier data partitions.
- Save split statistics and reusable split files.
- Begin graph construction using the encoded interactions and cleaned product metadata.


## 25 July 2026 - Preprocessing Validation

### Activities Completed

- Continued reviewing the Amazon Reviews 2023 literature and identified the official Electronics 5-core benchmark released by the dataset authors.
- Developed a dedicated validation notebook to compare my independently implemented preprocessing pipeline against the official benchmark.
- Compared aggregate statistics, user–item pairs, ratings, and timestamps between the two datasets.
- Performed additional verification against the original raw Amazon Electronics review data to investigate the observed discrepancies.

### Results

- Successfully reproduced the official benchmark in terms of:
  - Number of interactions
  - Number of users
  - Number of products
  - Number of unique user–item pairs
- Confirmed that there were **no missing or additional user–item interactions** compared with the official benchmark.
- Achieved approximately **99.83%** agreement in ratings and timestamps.
- Identified that the remaining **0.17%** of differences were caused solely by the duplicate-review retention policy.
- Verified using the raw Amazon review data that:
  - The official benchmark retains the **earliest** duplicate review.
  - My preprocessing pipeline currently retains the **most recent** duplicate review.

### Reflection

The validation demonstrates that my independently implemented preprocessing pipeline successfully reproduces the official Amazon Reviews 2023 Electronics 5-core benchmark. The only remaining methodological decision is whether to retain the current duplicate-review policy or modify the preprocessing pipeline to match the official benchmark exactly.

### Research Decisions

- No changes to the preprocessing pipeline have been made yet.
- A decision regarding the duplicate-review retention policy will be taken after discussing the validation results with the project supervisor.

### Next Steps

- Contact my supervisor for guidance regarding the duplicate-review policy.
- Depending on the decision, either:
  - rerun the affected stages of the preprocessing pipeline using the earliest duplicate-review policy, or
  - continue with the current implementation.
- Begin preparing the train/validation/test split notebook while awaiting feedback.

## 26 July 2026 – Benchmark Reproduction

- Literature Review
- Modified the preprocessing pipeline to retain the **earliest** duplicate review instead of the most recent.
- Reran all preprocessing stages after duplicate resolution, including metadata encoding and identifier encoding.
- Created separate `*_earliest` outputs to preserve the original preprocessing results.
- Validated the new dataset against the official Amazon Reviews 2023 Electronics 5-core benchmark.
- Achieved **15,473,535 / 15,473,536 (99.999994%)** matching interactions, with only one remaining duplicate-review tie case.

## 27 July 2026 – Temporal Train/Validation/Test Splitting

### Completed

- Created the `src/` package and implemented `src/splitting.py` for dataset splitting, integrity checks, and diagnostics.
- Implemented the leave-last-out and global timestamp splitting protocols.
- Developed `04_train_validation_test_split_earliest.ipynb` to generate, validate, and analyse both splits.
- Downloaded the official benchmark split files and compared the generated splits against them.
- Investigated all remaining discrepancies against the encoded interaction data to establish their cause.
- Generated and saved evaluation user samples, cold-item lists, and users without training history for both protocols.

### Design Decisions

- Moved the splitting logic into reusable functions within `src/`.
- Used a deterministic `(user_idx, timestamp, item_idx)` ordering to ensure reproducible splits when timestamps are tied.
- Implemented both splitting protocols to compare their impact on evaluation.
- Retained `rating` and `verified_purchase` for future experiments.
- Fixed the evaluation sample at 50,000 users per protocol under a fixed seed. Users are sampled but items never are, following Krichene and Rendle (2020), who show sampled item metrics are inconsistent with their exact versions.
- Excluded users without training history from evaluation.

### Key Findings

- The global timestamp split reproduced the official benchmark exactly: 13,096,427 / 1,161,644 / 1,215,465 interactions, with no differing user–item pairs.
- The leave-last-out split differed for only 25 users (0.0015%), with all differences explained by timestamp ties.
- The discrepancy counts are internally consistent: 10 test-partition and 15 training-partition differences sum to the 25 validation-partition differences, as expected from the two possible tie configurations.
- Timestamp ties affected only 44 users (0.0027% of all users).
- Under leave-last-out, 99.9998% of test instances have training interactions that postdate them, averaging 19.03% of the training set. The global timestamp split eliminates this entirely (zero affected instances).
- The timestamp protocol exposes 36,915 strictly cold items carrying 543,250 test interactions, against 1,172 under leave-last-out — a large enough population to support a dedicated cold-start evaluation.
- The two protocols evaluate different populations: leave-last-out covers all 1,641,026 users, while the timestamp split covers 469,838 (28.63%) and leaves 36,208 users without training history.
- All integrity checks passed successfully.

## 28 July 2026 – Nested Dataset Subsets

### Completed

- Added `src/subsets.py` for user sampling, iterative 5-core filtering, and re-encoding.
- Developed `05_nested_dataset_subsets.ipynb` to build the small/medium/full scalability ladder.
- Ran a calibration step over six sampling fractions to select the subset sizes from measured interaction counts.
- Built three nested subsets, generated both splitting protocols, and saved the interaction data, mappings, splits, cold-start lists, and evaluation samples.

### Design Decisions

- Sampled users as prefixes of a single shuffled permutation to guarantee nested subsets.
- Re-applied 5-core filtering after sampling to preserve the minimum interaction constraint.
- Re-encoded each subset to contiguous indices while preserving mappings back to the global indices for later metadata joins.
- Kept the original timestamp cutoffs unchanged across all subset levels.
- Built the `full` subset through the same pipeline so every level shares an identical directory structure.

### Key Findings

- The scalability ladder contains 2,346,776 / 6,297,459 / 15,473,536 interactions, giving near-geometric spacing between the three dataset sizes.
- Mean user degree remained almost constant across the subsets, while mean item degree decreased noticeably, confirming that the subsets are not uniformly scaled versions of the full graph.
- The subsets are exactly nested, with no users, items or interactions appearing in a smaller subset that are absent from a larger one.
- Temporal leakage depended on the splitting protocol rather than dataset size. Leave-last-out showed almost identical leakage across all subsets, whereas the timestamp protocol eliminated it completely.
- The `small` timestamp split contains 8,353 strictly cold items with 64,088 test interactions, providing sufficient data for the planned cold-start experiments.
- The `full` subset reproduced the Notebook 04 split counts exactly.
- All 32 integrity checks passed.

### Next Steps

- Build the bipartite user-item graph from the training splits.
- Derive item features from the cleaned metadata using the item mappings.
- Implement the evaluation pipeline and validate it with a popularity baseline before training the recommendation models.