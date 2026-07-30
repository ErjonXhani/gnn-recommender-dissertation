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

## 28 July 2026 – Graph Construction

### Completed

- Added `src/graph.py` for constructing and validating the user–item graph.
- Created `06_graph_construction.ipynb` to build graphs for all subsets under both splitting protocols.
- Installed PyTorch with CUDA 12.8 and verified GPU support for the RTX 5070 Ti.
- Confirmed that the graph loader reproduces the notebook graphs exactly.

### Design Decisions

- Used a single node index space for users and items.
- Represented each interaction as bidirectional edges.
- Applied LightGCN's symmetric normalization (`D^-1/2 A D^-1/2`) without self-loops.
- Rebuilt graphs on demand instead of storing derived graph files.
- Used the spectral radius as a correctness check for the normalized adjacency.
- Used adaptive power iteration until convergence.

### Key Findings

- No validation or test interactions leaked into any training graph.
- All adjacency matrices were symmetric and passed consistency checks.
- All graphs reached the expected spectral radius of **1.000000**.
- Leave-last-out produced exactly one giant component plus isolated nodes and nothing in between: 427 components for small (1 + 426 singletons), 693 for medium, 1,177 for full.
- Timestamp graphs remained approximately **96% connected**, with most remaining nodes being expected cold users and cold items.
- Graph sizes are manageable on the local GPU: small/last_out is 363,895 nodes and 3.66M directed edges, full/timestamp is 2.0M nodes and 26.2M.
- All **54 validation checks** passed successfully.


## 29 July 2026 – Item Features

### Completed

- Added `src/features.py` for text preparation, encoding, feature assembly, and loading.
- Developed `07_item_features.ipynb` to convert the cleaned metadata into numeric item features.
- Audited the metadata for post-cutoff fields and removed those that could introduce temporal leakage.
- Encoded all items using BLaIR, MiniLM, and TF-IDF+SVD.
- Built a one-hot encoding of `main_category` and assembled the final feature matrices.
- Installed `sentence-transformers` and `scikit-learn`.

### Design Decisions

- Selected BLaIR as the primary encoder because it was pre-trained by the dataset authors on Amazon Reviews 2023 item metadata. MiniLM and TF-IDF+SVD are used as comparison baselines for the cold-start experiments.
- Followed the BLaIR model card by using CLS pooling with L2 normalisation.
- Selected a 512-token limit for BLaIR after analysing the metadata token distribution.
- Used MiniLM with its native 256-token input length.
- Removed `Best Sellers Rank` before encoding to prevent temporal leakage while retaining `Date First Available`.
- Represented `store` through the text embeddings rather than as a separate categorical feature, while one-hot encoding `main_category`.
- Centred and re-normalised the text embeddings before combining them with the categorical features.

### Key Findings

- `Best Sellers Rank` was present in **56.3%** of items and was successfully removed before encoding.
- Metadata length exceeded the encoder limit for many items, confirming that truncation is necessary. At a 512-token limit, **67.5%** of items are encoded without truncation.
- BLaIR produced coherent item neighbourhoods, with sampled nearest neighbours consistently representing similar product types.
- Centring substantially improved the embedding distribution, reducing the mean pairwise cosine similarity from **0.645** to **0.003**.
- TF-IDF+SVD retained **26.4%** of the original TF-IDF variance using 256 components.
- `store` contains **49,344** distinct values, making it unsuitable as a separate categorical feature, while `main_category` contains only **39** levels and was included as a one-hot feature.
- Final feature dimensions were **807** (BLaIR), **423** (MiniLM), and **295** (TF-IDF+SVD).
- All validation checks passed, including feature alignment, cold-item coverage, and reconstruction of all subset feature matrices.


## 30 July 2026 – Evaluation and Baseline Models

### Completed

- Added `src/evaluation.py`, `src/baselines.py`, `src/models.py` and `src/training.py`.
- Developed `08_evaluation_and_baselines.ipynb` covering the evaluation pipeline and the three non-GNN baselines.
- Verified the evaluator against hand-computed metrics, a random recommender, and an independent implementation before running any model.
- Implemented and evaluated Popularity, ItemKNN and BPR-MF under both splitting protocols.
- Tuned ItemKNN shrinkage and BPR-MF dimension and regularisation on validation data.

### Design Decisions

- Wrote and verified the evaluation code before implementing any model. A faulty evaluator does not raise an error, it returns a plausible but incorrect number, so it needed checking against something whose behaviour could be reasoned about independently.
- Ranked every evaluation user against the complete item catalogue. Users are sampled, items never are, following Krichene and Rendle (2020), who show sampled item metrics are inconsistent with their exact versions and can reverse the ordering of models.
- Reported HitRate@K and NDCG@K under leave-last-out, where Recall@K is identical to HitRate@K because each user has exactly one relevant item, and Recall@K under the timestamp protocol, where users average 2.10 relevant items.
- Masked training items always, and validation items as well when evaluating on test, since the user has already consumed them by that point.
- Selected all hyperparameters on validation data only. Test was evaluated once per model with the chosen configuration.
- Used a 10,000-user validation subset for hyperparameter search and the full 50,000 users for early stopping and reported results.
- Treated BPR-MF as also serving the role of 0-layer LightGCN, since LightGCN without propagation is exactly its embedding table trained with BPR. Notebook 09 will use the same loss, sampler, dimension and optimiser, making the graph comparison controlled.
- Applied L2 regularisation explicitly to the embeddings in each batch rather than through the optimiser's weight decay, which would shrink every embedding on every step including those not involved in the batch.
- Sampled negatives uniformly without rejecting known positives; with 7.08 training interactions per user out of 105,298 items the collision rate is below 0.01%.

### Key Findings

- **Model ranking reverses between the two evaluation protocols.** Under leave-last-out the order by NDCG@10 is ItemKNN (0.00948), Popularity (0.00752), BPR-MF (0.00576). Under the global timestamp split it is BPR-MF (0.00223), ItemKNN (0.00217), Popularity (0.00124). BPR-MF moves from last to first and Popularity from second to last. This is the effect described by Ji et al. (2023), observed directly rather than cited.
- Popularity's NDCG@10 falls from 0.00752 to 0.00124 between protocols. Under leave-last-out its item counts include activity occurring after many test instances, so it is partly reading the future; under the timestamp split it is computed honestly and collapses.
- ItemKNN with plain cosine similarity scored roughly a third of Popularity (HR@10 0.00418 against 0.01510). The diagnostic showed why: the median training count of a recommended item was 5, the minimum possible under 5-core filtering, against 3,136 for Popularity. Cosine's denominator penalises popular items so heavily that pairs supported by two users outrank pairs supported by hundreds.
- Adding shrinkage to the similarity denominator, selected on validation, raised ItemKNN to HR@10 0.01736 and NDCG@10 0.00948, above Popularity on both protocols. Selected values were 200 for leave-last-out and 500 for timestamp.
- BPR-MF at the LightGCN paper's default settings (dimension 64, regularisation 1e-4) overfitted immediately: training loss fell to 0.011 while validation NDCG declined from the first evaluation onwards. The model has 23.3M parameters fitted to 1.83M training interactions, about 12.7 parameters per observation.
- Tuning recovered most of the gap. Dimension 32 with regularisation 0.05 (leave-last-out) and 0.08 (timestamp) was selected on validation. Regularisation above roughly 0.2 collapses the model entirely, producing results at random level.
- BPR-MF still does not beat Popularity under leave-last-out (NDCG@10 0.00576 against 0.00752) but does under the timestamp protocol (0.00223 against 0.00124).
- Literature default hyperparameters were unsuitable for this data in both ItemKNN and BPR-MF. Both were tuned on considerably denser datasets: Amazon-Book has 56.7 interactions per user and density 6.2e-4, against 9.1 and 8.6e-5 here.
- Absolute metric values are low by design. Ranking is against the full catalogue of 105,298 items rather than a sampled subset, which Ji et al. note produces values much lower than sampled-metric results. Popularity nonetheless achieves 159 times the random hit rate.
- All automated checks passed, including the evaluator agreeing exactly with an independent implementation on a 2,000-user sample.

### Next Steps

- Implement LightGCN, NGCF and GraphSAGE using the same training loop and evaluator.
- Verify the LightGCN implementation against published Gowalla or Amazon-Book results before trusting its numbers on this dataset.
- Consider whether the protocol-dependent ranking reversal warrants its own results subsection.