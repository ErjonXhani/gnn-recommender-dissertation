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

### Discussion (09 Aug 26)
- **BLaIR's pretraining is a potential leakage route, and the encoder ablation rules it out.** BLaIR was released by the dataset authors alongside Amazon Reviews 2023 (Hou et al., arXiv:2403.03952) and pretrained on review–metadata pairs from the whole dataset, including the period held out for testing. Its weights could therefore in principle encode something about the test window that a general-purpose encoder could not. The encoder comparison in Notebook 11 addresses this directly: MiniLM, trained on general web text with no exposure to Amazon data, scores 0.00873 against BLaIR's 0.00872. If BLaIR's pretraining conferred any test-period knowledge it would appear as an advantage over MiniLM, and it does not.

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

## 31 July 2026 – Graph Model Implementation and LightGCN Reproduction

### Completed

- Added `GraphRecommender`, `LightGCN`, `NGCF` and `GraphSAGE` to `src/models.py`, all sharing one interface with `MatrixFactorization`.
- Added `mean_adjacency` to `src/graph.py` for GraphSAGE's row-normalised aggregation.
- Extended `src/training.py` with a `sampling` option, moved L2 regularisation onto the layer-0 embeddings, and fixed the history record being dropped when early stopping fires.
- Developed `09_model_implementation_and_reproduction.ipynb`: implementation verification followed by a reproduction of LightGCN on the authors' released Gowalla split.
- Ran three seeds at the published configuration, plus one diagnostic run isolating the sampling scheme.

### Design Decisions

- Verified the implementation before running it on the dissertation data. Internal consistency is not correctness, so the check had to be against published numbers on data where the answer is known.
- Used the 0-layer identity as the primary structural test: LightGCN without propagation is exactly matrix factorisation, so with the same seed it must produce identical embeddings and identical training results.
- Recomputed one node's layer-1 embedding by hand from its neighbours to confirm the sparse multiplication implements the stated formula rather than something merely plausible.
- Initialised the item feature projection at zero so that a model built with metadata starts exactly equal to its feature-free counterpart. The metadata ablation is then a strict nesting rather than a comparison of two different models.
- Regularised the layer-0 embeddings rather than the propagated ones, matching the LightGCN and NGCF reference implementations. These are the only free parameters; for matrix factorisation the two are identical, so Notebook 08 is unaffected.
- Followed the paper's protocol for the reproduction only. The released Gowalla data has no validation set and the authors select the best epoch on test, so the final-epoch value is reported as the headline with the best-epoch value stated separately. All experiments on the Amazon data use a proper validation split.
- Adopted user-uniform sampling for every model in all later experiments, chosen for consistency with the reference implementations and alignment with per-user-averaged metrics, not because it scores higher.
- Converted the adjacency to CSR, which halved training time (7.46s to 3.77s per epoch) at identical arithmetic.

### Key Findings

- **LightGCN reproduces.** Three seeds give recall@20 0.18184 ± 0.00004 and NDCG@20 0.15542 ± 0.00041, against published values of 0.1823 and 0.1555. NDCG is within seed variance; recall is 0.25% low.
- **The sampling scheme accounts for most of the initial gap, and it is not described in the paper.** LightGCN's released code draws a user uniformly and then one of that user's items; iterating over interactions instead gives recall@20 0.1738 and NDCG@20 0.1482, roughly 4.6% below published on both metrics.
- The sampler is worth 4.7% of recall here, more than the 3.9% the paper reports for increasing propagation depth from one layer to three. An undocumented implementation detail therefore mattered more than the mechanism the paper is about.
- Part of the remaining 0.25% is estimator choice. The paper reports the best of roughly 100 test evaluations while these runs report the final epoch; on seed 1 those two estimators differ by 0.00024.
- The 0-layer identity holds exactly at initialisation and to 1e-7 after five epochs. Both models are individually deterministic, so the residual is the order in which the loss and regularisation gradients are summed, not nondeterminism.
- Parameter counts confirm the architectures: LightGCN has 11,644,640 parameters at dimension 32, exactly the two embedding tables and nothing else; NGCF adds 6,336 and GraphSAGE 4,160.
- Propagation matches the definition to 7.45e-09 when one node's layer-1 embedding is recomputed by hand from its neighbours.
- Published λ = 1e-4 corresponds to 5e-5 in this codebase, because the reference implementation carries a factor of ½ in the norm.
- All checks passed.


## 31 July 2026 – Sampler Change Applied to Notebook 08

### Completed

- Re-ran all trained baselines in `08_evaluation_and_baselines.ipynb` under user-uniform sampling, following the finding in Notebook 09.
- Replaced the leave-last-out regularisation selection with a single consistent sweep, since the earlier values came from sweeps using different evaluators and epoch budgets and were not comparable to each other.
- Extended the timestamp regularisation grid, which previously selected the boundary value.

### Design Decisions

- Applied user-uniform sampling to BPR-MF as well as the graph models. BPR-MF is 0-layer LightGCN and serves as the controlled comparison, so a different sampler would confound propagation with sampling and would understate the baseline in the direction that flatters the graph models.
- Left Popularity, ItemKNN and Random untouched; they do not train, so no sampler applies.
- Kept the exploratory sweeps in the notebook as the record of how the range was narrowed, and added a final sweep with a fixed evaluator, epoch budget and patience for the actual selection.

### Key Findings

- The sampler is worth considerably more on this data than on Gowalla. BPR-MF NDCG@10 rose from 0.00576 to 0.00693 under leave-last-out (+20%) and from 0.00223 to 0.00497 under the timestamp protocol (+123%).
- The gain is largest where user histories are thinnest, consistent with user-uniform sampling giving low-degree users the same number of updates as heavy ones.
- Selected regularisation increased on both protocols: 0.05 to 0.10 (leave-last-out) and 0.08 to 0.15 (timestamp).
- Leave-last-out performance is flat across regularisation 0.05 to 0.12, varying by about 11% non-monotonically, so 0.10 is a selection rather than a distinguishable optimum. The timestamp sweep has a clear interior maximum at 0.15.
- The protocol reversal survives, but is narrower. BPR-MF now reaches 92% of Popularity's NDCG@10 under leave-last-out, against 77% before.
- Popularity and ItemKNN results reproduced exactly, confirming that only the sampler changed.
- The paper-default configuration still overfits immediately: validation NDCG declines from the first evaluation onwards under both protocols.
- All 28 checks passed.


## 1–4 August 2026 – Notebook 10: Tuning, Comparison, and NGCF Reproduction

### Completed

- Built `10_hyperparameter_tuning.ipynb`: configuration, per-protocol graphs and adjacencies, and three evaluators (test, validation, and a 10,000-user tuning subset).
- Built a caching tuning harness that writes each run's history to `results/10_tuning/` and skips configurations already completed, so an interrupted session loses only the run in progress. 50 tuning runs in total.
- Tuned LightGCN, NGCF and GraphSAGE on both protocols, and diagnosed why GraphSAGE could not train at all in its published form.
- Added a `final_activation` switch to `GraphSAGE` and tested whether the output nonlinearity helps or hurts.
- Added Section 6 to `09_models_and_verification.ipynb`: an external check of the NGCF implementation against published Gowalla results.
- Ran the final comparison: 28 runs across four models, two protocols and multiple seeds, with per-user scores saved for the significance tests in Notebook 12.
- Diagnosed and corrected a checkpoint-selection problem that was understating BPR-MF by roughly 30%.

### Design Decisions

- Built a separate graph per protocol from that protocol's training set, and asserted that no validation or test interaction appears as an edge. Propagation over held-out edges would leak the answer without changing anything visible in the metrics.
- Stored both adjacencies in CSR format, which halved epoch time relative to COO at identical arithmetic.
- Fixed the epoch budget at 300 for every configuration. A uniform budget matters more than the exact value, since models converge at different rates and per-model budgets would confound convergence speed with model quality.
- Kept the learning rate at 1e-3. Tested 3e-3, which converged only about 1.5 times faster and gave a visibly noisier validation curve; not worth introducing a difference from the BPR-MF baseline.
- Used a 10,000-user validation subset for the grid search, reserving the full evaluation set for reported results.
- Tuned each model by coordinate descent, except LightGCN where depth and regularisation were searched jointly after they proved to interact.
- Used BPR-MF at its own tuned regularisation as the depth-0 anchor, rather than LightGCN with zero layers at the graph model's regularisation, which would have been a straw man.
- Kept `normalise_output` and `final_activation` defaulting to the paper's behaviour in `models.py`, passing the chosen values explicitly at the call site so the deviation is visible in the notebook rather than buried in a class default.
- Stopped tuning GraphSAGE once the variation within a single run exceeded the differences between configurations. Further search would have been fitting noise.
- Re-tuned under the timestamp protocol rather than transferring the leave-last-out settings, since Notebook 08 had already shown the two protocols select different regularisation for the same model.
- Scaled checkpoint granularity to each model's convergence timescale: BPR-MF peaks around epoch 20 and is evaluated every 2 epochs, the graph models peak past 250 and are evaluated every 10. Patience is held equal in epochs at 100, so all four models receive the same tolerance.
- Ran BPR-MF with five seeds against the graph models' three, because its seed variance is an order of magnitude larger and it carries the comparison the chapter rests on.
- Verified NGCF against published results before reporting anything about it, because the Amazon results contain a negative claim about a published method, and such a claim is only as strong as the implementation behind it.
- Keyed the cache on model, protocol, depth, regularisation, epoch budget, seed and any architectural overrides, so runs at different settings cannot silently share a file.

### Key Findings: tuning methodology

- **Graph models need far weaker regularisation than matrix factorisation.** LightGCN's optimum is 3e-3 against BPR-MF's 0.10. Propagation averages each embedding with its neighbours and is itself a regulariser, and the loss gradient reaching a layer-0 embedding is diluted through the propagation layers while the L2 penalty applies directly. The first grid, inherited from BPR-MF's range, sat entirely in a region where the graph models cannot train.
- **Comparing hyperparameters at a fixed epoch budget confounds regularisation strength with convergence speed.** Stronger regularisation converges more slowly but reaches higher: at 150 epochs the surface appeared flat, but at 300 epochs 3e-4 peaked at epoch 120 and then declined while 1e-3 was still improving at epoch 230.
- **Depth and regularisation interact, and optimal regularisation falls monotonically with depth**: 1e-2, 3e-3, 3e-3, 1e-3 for one to four layers. Deeper models receive more smoothing from propagation and need less explicit shrinkage.
- **Checkpoint granularity has to be scaled to a model's convergence timescale.** Evaluating BPR-MF every 10 epochs instead of every 2 selected a checkpoint around epoch 70 that scored marginally higher on validation and 30% worse on test. Correcting it reduced LightGCN's leave-last-out advantage from 67% to 27% and widened BPR-MF's timestamp lead; both corrections moved against the hypothesis being tested.
- A 500-epoch probe showed no reliable improvement past epoch 300. The apparent peak at 500 was the maximum of twenty-one noisy evaluations on a flat plateau, about two standard deviations above the plateau mean - the same maximisation bias identified in LightGCN's published Gowalla figure.
- Cost per epoch on this data: LightGCN 7.07s, NGCF 11.42s, GraphSAGE 6.87s. Sparse propagation is far more expensive here than arithmetic alone predicts, because the graph averages 10 neighbours per node against Gowalla's 23 and short rows multiply per-row overhead.

### Key Findings: model behaviour

- **GraphSAGE could not train in its published form.** Training loss moved only from 0.558 to 0.447 over 300 epochs, against LightGCN's 0.682 to 0.051. The cause is structural: ReLU on the final layer confines embeddings to the non-negative orthant and L2 normalisation projects them onto the unit sphere, so every score is a cosine similarity bounded in [0, 1] and BPR cannot open a margin wider than 1. Disabling the output normalisation raised validation NDCG@10 from 0.000976 to 0.006032, a sixfold improvement.
- **Removing the final nonlinearity made GraphSAGE worse, not better** - 0.004750 against 0.006393 at the same regularisation, a 26% loss. This is mild counter-evidence to LightGCN's claim that nonlinear activation is harmful in collaborative filtering: the claim does not generalise across architectures.
- GraphSAGE overfits rather than underfits. Its final training loss (0.031) is lower than LightGCN's (0.051) while its validation score is 40% worse, and it peaks around epoch 30 before declining - the opposite failure mode to the one LightGCN's authors report for NGCF. Its validation curve is also noisy enough that selection is unreliable: within one run the score swings between 0.0051 and 0.0078, a band as wide as the entire spread across its three regularisation values.
- **Most of the benefit of propagation comes from the first hop.** With regularisation tuned separately at each depth, one layer gives +27.3% over BPR-MF, three give +33.7%, and four fall back to +29.2%. A fixed-regularisation sweep understated the one-layer result by 11.6% and made depth appear far more important than it is.
- **All three models peak at three propagation layers**, independently and on very different absolute scales.
- **NGCF reproduces on Gowalla**: recall@20 0.15349 and NDCG@20 0.12224 against published 0.1569 and 0.1327, using the same graph, evaluator and training loop as the LightGCN reproduction. Both figures are lower bounds, since the selected regularisation sat at the top of the searched range and the run was still improving at epoch 400.
- Recall reproduces more closely than NDCG for NGCF (2.2% against 7.9%), unlike LightGCN where both metrics were off by the same 0.25%. Retrieving approximately the right items but ordering them slightly worse points to a small residual difference, most plausibly dropout: this implementation applies a single message dropout at 0.1 where NGCF uses both message and node dropout.
- NGCF is considerably more expensive than LightGCN on the same graph - roughly 15s per epoch on Gowalla at dimension 64 against LightGCN's 3.8s - owing to two weight matrices and an elementwise interaction term per layer.

### Key Findings: results

- Selected configurations, with validation NDCG@10 under leave-last-out: LightGCN 0.012740 (3 layers, reg 3e-3), BPR-MF 0.009530, NGCF 0.007964 (3 layers, reg 1e-2), GraphSAGE 0.007793 (3 layers, reg 3e-3, output normalisation disabled).
- **Under leave-last-out, LightGCN improves on matrix factorisation by 27%** on test: NDCG@10 0.00812 ± 0.00007 against 0.00641 ± 0.00069, with no overlap between the ranges. NGCF and GraphSAGE both fall below the baseline at 0.00425.
- **LightGCN does not beat ItemKNN** (0.00948), so the best-performing method under this protocol is item-item neighbourhood collaborative filtering rather than any graph neural network. This does not generalise: on Gowalla, ItemKNN comes last and LightGCN beats it by 26%.
- **The ranking reverses completely under the timestamp protocol.** BPR-MF reaches 0.00465 ± 0.00040 against LightGCN's 0.00158 ± 0.00020, a factor of 2.9. Every graph model also falls below ItemKNN, and LightGCN barely exceeds Popularity.
- The reversal is not an artefact of the test population. It is already visible on validation, where all models are scored on identical users: BPR-MF 0.0089 against LightGCN 0.0049. Validation-to-test ratios are roughly uniform within a protocol - 1.5 to 1.7 under leave-last-out, 2.0 to 3.1 under timestamp - so the larger drop under the temporal split reflects a harder and differently constituted evaluation population rather than model-specific failure.
- **Both graph models other than LightGCN underperform matrix factorisation on both protocols.** This is stronger than the result LightGCN's authors report, where NGCF beats matrix factorisation on Gowalla. The same NGCF implementation reaches 98% of published recall there and loses here, so the failure belongs to the data regime rather than the code. The mechanism is examined separately below.
- The ranking is not an artefact of the epoch budget. Several configurations of NGCF and GraphSAGE hit the 300-epoch ceiling, but the selected configuration of each converged inside it, NGCF peaking at epoch 230 and GraphSAGE at epoch 30.
- Seed variance differs sharply by model: LightGCN 0.9% of its mean against BPR-MF's 14%. Graph propagation appears to stabilise the solution as well as improve it under leave-last-out.
- The ordering is identical under NDCG@10, HitRate@10 and Recall@20, so no conclusion depends on the choice of metric.

## 5 August 2026 – Matrix Factorisation on Gowalla: Density Reference and Baseline Audit

### Completed

- Added Section 7 to `09_models_and_verification.ipynb`: matrix factorisation trained on Gowalla using the same configuration, evaluator and protocol as the LightGCN and NGCF reproductions, with its own regularisation sweep.
- Fact-checked every published figure used in Notebook 09 against the source papers.

### Sources Verified

- The Gowalla split originates with NGCF (Wang et al., SIGIR 2019, arXiv:1905.08108), not LightGCN, which reused the same files. NGCF Section 4.1: 10-core filtering, then 80% of each user's interactions sampled at random for training and the remainder for test, with 10% of training held out for validation. **The split is random, not temporal.**
- Dataset statistics 29,858 users / 40,981 items / 1,027,370 interactions / density 0.00084 (NGCF Table 1, LightGCN Table 2) match the loaded files exactly.
- Published figures: MF Recall@20 0.1291 and NDCG@20 0.1109 (NGCF Table 2); NGCF-3 0.1569 and 0.1327 (NGCF Tables 2 and 3); LightGCN-3 0.1823 and 0.1555 (He et al., Table 3).
- NGCF describes its MF baseline as matrix factorisation optimised with the BPR loss, so it is directly comparable to the BPR-MF used throughout this project.

### Design Decisions

- Gave matrix factorisation its own regularisation sweep rather than inheriting LightGCN's. On the Amazon data its optimum was roughly five hundred times larger, so transferring the graph model's setting would have produced a straw man. The Gowalla optimum is 1e-3, an interior maximum.
- Held everything else identical to the graph model runs - dimension 64, batch 2048, user-uniform sampling, 1000 epochs, best-epoch selection - so the only difference between arms is the model.

### Key Findings

- **Recall@20 on Gowalla: LightGCN 0.18184, BPR-MF 0.15515, NGCF 0.15349.** LightGCN beats matrix factorisation by 17.2%; NGCF is 1.1% below it.
- **This contradicts the sparsity explanation previously offered for the Amazon results.** LightGCN's advantage over matrix factorisation is 17.2% on Gowalla, where users hold 27 training interactions, and 26.7% on Amazon Electronics under leave-last-out, where they hold 7. The advantage is larger on the sparser data, so density does not account for the collapse under the timestamp protocol.
- **What separates the three cases is temporal structure in the evaluation.** Gowalla's split is random, with no temporal shift. Amazon leave-last-out orders each user's interactions but leaves the training graph containing activity that postdates many users' test items. The Amazon timestamp split has a genuine global gap and no leakage. LightGCN gains 17.2%, then 26.7%, then loses by a factor of 2.9. The working hypothesis is that graph propagation benefits disproportionately from temporal leakage: propagation aggregates a neighbourhood encoding what was co-consumed during the training window, and where that window overlaps the test period the neighbourhood carries information about the future it should not have.
- **The properly tuned baseline beats the published one by 20%.** BPR-MF here reaches Recall@20 0.15515 against the 0.1291 reported for MF on the same files. Consequently NGCF's published figure of 0.1569 is only **1.1%** above this baseline, where its paper claims an 11.68% improvement over its own MF; and LightGCN's published 0.1823 is **17.5%** above it, rather than the 41% implied by the published MF number.
- The gain over the published baseline comes from two sources already documented in this project: user-uniform sampling, worth about 4.7% on this dataset, and a regularisation sweep with an interior optimum at 1e-3.
- This is the Rendle et al. and Ferrari Dacrema et al. critique of under-tuned baselines in recommender research, reproduced as a direct measurement on the benchmark's own data using code verified to reproduce both papers.

- **ItemKNN comes last on Gowalla** at Recall@20 0.14396, below matrix factorisation (0.15515) and 26% below LightGCN (0.18184). On the Amazon data under leave-last-out it comes first. The ordering of methods is therefore almost completely different between the two datasets.
- **This narrows the ItemKNN finding.** The claim is not that a neighbourhood method is generally competitive with graph models - on the GNN literature's own benchmark it is not. The claim is that on sparse e-commerce data under leave-last-out, a well-tuned ItemKNN outperforms every graph model tested.
- **The methods that win under each protocol share a pattern.** ItemKNN and LightGCN both rely on co-occurrence structure, one explicitly and one through propagation, and both peak under the leaky leave-last-out protocol before falling under the leakage-free timestamp split. Matrix factorisation, which learns coarse latent factors under heavy regularisation, wins where there is a genuine temporal gap. This explains all three protocols with one mechanism and is not specific to graph models.
- Two secondary confirmations: shrinkage 0 gives 0.11569 against 0.14396 at shrinkage 50, the same cosine bias towards rare items diagnosed on the Amazon data; and the optimal shrinkage is 50 here against 200 to 500 on Amazon, since denser data needs less correction.

### Caveats

- The leakage hypothesis remains a hypothesis. Dataset and protocol vary together across the three cases, and the clean test - Gowalla under a temporal split - is not possible with the released files.
- The baseline comparison uses best-epoch selection over 1000 epochs, while NGCF used validation-based early stopping. That favours this implementation, though the same analysis on LightGCN put the estimator bias at roughly 0.2%, not 20%.
- NGCF re-implemented the NDCG metric, so Recall@20 is the safer basis for comparison against its published table.
- The Gowalla NGCF and MF results are single-seed. LightGCN is three seeds.


## 5-7 August 2026 – Notebook 11: Metadata and Cold-Start Recommendation

### Completed

- Built `11_metadata_and_cold_start.ipynb`, reusing Notebook 10's splits, graphs, evaluators and training loop unchanged so the feature and no-feature arms are comparable without qualification.
- Loaded the three encoders built in Notebook 07 and confirmed every item has a non-empty feature vector.
- Ran regularisation checks with features on both protocols, then the main comparison: LightGCN with and without BLaIR, both protocols, three seeds, on test.
- Compared BLaIR against MiniLM and TF-IDF+SVD on leave-last-out.
- Repeated the comparison for matrix factorisation, implemented as LightGCN with zero propagation layers.
- Trained GraphSAGE with features on leave-last-out.
- Evaluated cold items under two framings: open ranking against the full catalogue, and ranking restricted to cold items only.

### Design Decisions

- Carried only LightGCN into the main comparison. NGCF and GraphSAGE both lost to matrix factorisation in Notebook 10, so adding features to them would be improving models that are already behind.
- Reused Notebook 10's feature-free results rather than retraining them. The feature projection is initialised at zero, so the two models start numerically identical and the no-feature arm *is* the feature model with the projection unused.
- Gave every arm its own regularisation sweep. This proved essential rather than pedantic: features moved LightGCN's optimum from 3e-3 to 1e-4, and using the old value would have reported a 2% gain instead of the true 15% on validation.
- Implemented matrix factorisation with features as LightGCN with `n_layers=0`, which Notebook 09 verified is exactly matrix factorisation. This reused tested code rather than adding a feature hook to a second class.
- Matched checkpoint granularity per model, as in Notebook 10: every 2 epochs for matrix factorisation, every 10 for the graph models, patience held equal in epochs.
- Ran the cold-start evaluation on the timestamp protocol only. Under leave-last-out just 0.7% of evaluation users encounter a cold item, too few to measure anything.
- Evaluated cold items two ways, because the questions differ: whether a new product can appear in an unrestricted top-10 list, and whether new products can be ranked sensibly among themselves.
- Left the encoder ablation at one seed and reasoned about its significance from the seed variance measured for BLaIR in the main comparison, rather than spending two hours on a supporting result.
- Deferred a content-based ItemKNN to future work. Metadata plugs into matrix factorisation for free because both represent an item as a vector; ItemKNN learns nothing and stores a similarity table, so using metadata there means designing a new hybrid model rather than running an ablation.
- Saved model weights for the cold-start runs, correcting an omission in Notebooks 10 and 11 where only metrics were kept and models had to be retrained to be re-scored.

### Key Findings: metadata

- **Metadata helps the graph model and hurts matrix factorisation.** LightGCN improves from 0.00812 to 0.00864 under leave-last-out (+6.3%) and from 0.00158 to 0.00230 under timestamp (+45.9%). Matrix factorisation falls from 0.00641 to 0.00574 (−10.5%) and from 0.00465 to 0.00422 (−9.2%). The matrix factorisation comparisons use identical regularisation in both arms, so this is not a tuning artefact. The significance testing in Notebook 12 later qualified the timestamp result: consistent in direction across seeds, but reaching significance in only one of three.
- **Content and propagation are complementary rather than interchangeable.** Propagation smooths a feature-derived representation across the graph neighbourhood, where it reinforces genuine co-purchase structure; without propagation the projection is extra capacity the model overfits. The tuning behaviour agrees: adding features moved LightGCN's optimal regularisation down thirtyfold, while matrix factorisation's optimum did not move at all.
- **The benefit is largest where the collaborative signal is weakest** - seven times larger under the timestamp protocol than under leave-last-out. It does not, however, overturn Notebook 10: matrix factorisation still wins the leakage-free protocol with or without metadata.
- Features gave matrix factorisation a 13% gain on validation and a 23.5% loss on test, the same validation-test divergence that caused the checkpoint problem in Notebook 10, and for the same reason: under leave-last-out, validation is each user's second-to-last item and test is their last one, so a weakly regularised model that memorises recent history scores well on one and badly on the other.

### Key Findings: encoders

- **The encoder does not matter.** BLaIR 0.00872, MiniLM 0.00873, TF-IDF+SVD 0.00857 - a range of about 1.5 standard deviations of the seed variance measured for BLaIR. The domain-specific pre-training supplied with the dataset gives no measurable advantage over a general-purpose sentence encoder, and word counts with a matrix decomposition capture three quarters of the benefit.

### Key Findings: cold start

- **Leave-last-out barely has a cold-start problem**: 425 cold items affecting 338 users, 0.7% of the evaluation set, against 8,353 items affecting 22,130 users, 44.3%, under the timestamp protocol. This is a second way the standard protocol understates the difficulty of the real task, alongside the temporal leakage.
- **New products never appear in an unrestricted recommendation list.** Ranking across the full catalogue, every model scores at or below chance on cold items, metadata included. A cold item's embedding is never updated during training so it stays near its random initialisation, and warm items crowd it out. Surfacing new products requires a deliberate mechanism rather than better item representations.
- **Restricted to cold items, features matter and the model ranking inverts.** GraphSAGE, the weakest model overall at 0.00149, is the strongest on cold items at 0.00232 - five times random and twice LightGCN with the same features. LightGCN gains 34% from features here; matrix factorisation gains nothing, scoring barely above random with or without them.
- This is the property GraphSAGE was included to test, and it is the only setting in the project where it wins. Aggregating features into a representation beats projecting them onto an ID embedding where there is no interaction history to look up.

### Verification

- The cold-start models were retrained from scratch and reproduce their earlier scores exactly: LightGCN 0.001788, LightGCN with BLaIR 0.002368, matrix factorisation 0.004972, all matching Notebooks 10 and 11 to five decimal places.
- The matrix factorisation arm was built as LightGCN with zero layers, so this also re-confirms the zero-layer identity through a full training run in a third notebook.
- All 12 checks passed.


## 7-8 August 2026 – Notebook 12: Significance, Scaling and Synthesis

### Completed

- Built `12_significance_scaling_synthesis.ipynb`.
- Ran paired significance tests over the per-user scores saved by Notebooks 08, 10 and 11.
- Measured training cost per epoch for LightGCN and matrix factorisation on all three nested subsets.
- Ran a performance comparison of ItemKNN, LightGCN and matrix factorisation on `small` and `medium`, and matrix factorisation alone on `full`.
- Checked whether matrix factorisation's regularisation optimum shifts at larger scale.
- Assembled every test result from the project into one table, with figures for the cost scaling and the protocol inversion.

### Design Decisions

- Used the Wilcoxon signed-rank test rather than a paired t-test. Per-user NDCG is heavily zero-inflated and far from normal.
- Paired by user rather than comparing independent samples, since users differ enormously in how predictable they are and pairing removes that variance.
- Ran the tests on a single seed. Averaging per-user scores over different numbers of seeds spreads small non-zero values across more users and distorts the win counts; matrix factorisation has five seeds and the graph models three.
- Weighted effect sizes over p-values in interpretation. With 50,000 paired users almost any difference is significant, so the win counts and relative differences carry the information.
- Measured scaling on leave-last-out only, because it is the protocol where the three models sit closest together and so where scale has most room to change the ordering.
- Held hyperparameters at the values selected on `small` rather than re-tuning at each size, since tuning LightGCN on `medium` costs 3.6 hours per configuration. A partial check on matrix factorisation quantifies what this costs.
- Compared models only within a subset. Matrix factorisation on `full` is reported as a feasibility demonstration, not as a comparison against models trained on smaller data.
- Re-ran every model on `small` through the new code path rather than reusing earlier results, so all rows come from one pipeline and the earlier numbers are re-verified.

### Key Findings: significance

- **The two headline results are firm.** LightGCN beats matrix factorisation under leave-last-out (p = 5.8e-05) and matrix factorisation beats LightGCN under the timestamp protocol (p = 2.2e-41). ItemKNN beating LightGCN under leave-last-out is also significant (p = 2.4e-04).
- Metadata's benefit to LightGCN is significant on both protocols (p = 1.8e-03 and 1.5e-05).
- **Metadata's cost to matrix factorisation is significant under leave-last-out only** (−14.2%, p = 2.0e-04). Under the timestamp protocol the mean falls in all three seeds, by 4% to 9%, but the paired test reaches significance in only one. That finding should be stated as consistent in direction and weakly supported, not as established.
- **About 97% of users are tied in every comparison** - neither model retrieves anything the user wanted. Every reported difference is decided by the remaining 3%.

### Key Findings: scaling

- **LightGCN's training cost grows quadratically with dataset size**, fitted exponent 1.996 across subsets spanning 1.8 to 12.2 million interactions. Each batch propagates over the whole graph while the number of batches also grows with the data.
- **The cost gap to matrix factorisation widens with scale**: 13.8 times more wall-clock time on `small`, 22.1 times on `medium`. The per-epoch ratio understates this because LightGCN also requires more epochs.
- Matrix factorisation trains on the full 12.2 million interactions in 39 minutes. The same run for LightGCN would take roughly 25 hours, which is why the performance comparison stops at `medium`.
- Memory is not the constraint - peak usage was 5.4 GB on the full subset. Time is.
- **Accuracy falls for every model as the data grows**, because the catalogue grows with it and there are more candidates to rank against. Absolute scores are therefore not comparable across subsets.
- **The ordering is preserved from `small` to `medium` and the gaps widen.** LightGCN's advantage over matrix factorisation grows from 1.17x to 1.58x, and ItemKNN's over LightGCN from 1.17x to 1.25x.
- Matrix factorisation's regularisation optimum on `medium` is 0.03 rather than the 0.10 selected on `small`, worth 5.4%. Holding hyperparameters fixed therefore overstates the widening gap by roughly that much; corrected, LightGCN's advantage is 1.50x rather than 1.58x.

### Key Findings: synthesis

- **The ranking of models inverts almost completely between protocols.** Matrix factorisation moves from fifth to first, Popularity from fourth to last, ItemKNN from first to fourth.
- Popularity's collapse is the clearest signature of the leakage. Nothing about the model changed between protocols, only whether its item counts were allowed to include activity occurring after the test point.

### Verification

- The scaling study retrained every model on `small` from scratch and reproduced Notebooks 08 and 10 exactly: ItemKNN 0.00948, LightGCN 0.00811, matrix factorisation 0.00693.
- All 13 checks passed.

### Next Steps

- Experimental work is complete. Remaining effort goes to the dissertation document.
- Carry forward as future work: a second product category spanning a different density, a content-based ItemKNN, and per-size hyperparameter tuning for the scaling study.