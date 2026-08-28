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

- 50,000 evaluation users (10,000-user subset used for hyperparameter search only)
- 105,298 catalogue items
- 7.08 training interactions per user
- Full-catalogue evaluation (no item sampling)
- Training interactions per user: median 5, mean 7.08, 25th percentile 3, max 629
- 58.7% of users have 5 or fewer training interactions

**Model ranking by NDCG@10 reverses between protocols:**
- leave-last-out: ItemKNN 0.00948 > Popularity 0.00752 > BPR-MF 0.00693
- timestamp: BPR-MF 0.00497 > ItemKNN 0.00217 > Popularity 0.00124

**Sampling scheme (see Notebook 09):**
- BPR-MF NDCG@10, interaction-uniform to user-uniform:
  leave-last-out 0.00576 → 0.00693 (+20%), timestamp 0.00223 → 0.00497 (+123%)
- selected regularisation moved 0.05 → 0.10 (leave-last-out) and 0.08 → 0.15 (timestamp)

**Popularity across protocols:**
- NDCG@10: 0.00752 → 0.00124
- achieves 159x the random hit rate despite low absolute values

**ItemKNN:**
- HR@10 before and after shrinkage: 0.00418 → 0.01736
- median training count of recommended item: 5 → 280
- best shrinkage: 200 (leave-last-out), 500 (timestamp)

**BPR-MF:**
- parameters at default settings: 23.3M for 1.83M training interactions
  (12.7 per observation)
- best embedding dimension: 32; dimension had little effect (16/32/64 within 2% at
  fixed regularisation), so regularisation rather than capacity is the binding constraint
- validation NDCG@10 is flat across regularisation 0.05–0.12 under leave-last-out


# Notebook 09 - Model Implementation and Reproduction

### Implementation Verification

- Explain why implementation verification precedes any experiment: an incorrect model produces plausible numbers rather than an error, so correctness cannot be inferred from results.
- Present the three verification steps as complementary: the 0-layer identity tests everything except propagation, the hand-computed layer-1 embedding tests propagation itself, and reproduction against published results tests the whole pipeline end to end.
- Present the 0-layer identity as the primary structural test. LightGCN without propagation is exactly matrix factorisation, so the check is an equality rather than a plausibility argument.
- Report that parameter counts match the architectures exactly, confirming LightGCN carries no weights above its embedding tables.
- Explain the zero-initialised feature projection and why it makes the metadata ablation a strict nesting rather than a comparison of two different models.

### Reproduction Protocol

- State what constitutes a successful reproduction: matching published figures on the authors' own released split, using their configuration, across multiple seeds.
- Explain why Gowalla was chosen: the exact train/test files are public, so the split is not a source of disagreement. Attribute them correctly - the split originates with NGCF (Wang et al., 2019), which LightGCN reused. NGCF applies 10-core filtering and then samples 80% of each user's interactions at random for training, so the split is random rather than temporal. This matters for how the results are compared against the Amazon experiments.
- Note the protocol asymmetry. The released data has no validation set and the paper selects the best epoch on test; this notebook follows that protocol and states it explicitly, while all Amazon experiments use a proper validation split.
- Record the two configuration details a reader would need to reproduce this: the published lambda of 1e-4 corresponds to 5e-5 here, because the reference implementation carries a factor of one half in the regularisation term; and the batch size is 2048 rather than the paper's 1024 on Gowalla, which the result reproduces to within 0.25% regardless.

### LightGCN Reproduction

- Present the three-seed reproduction against published values.
- Present the sampler comparison and the training-curve figure, noting that the two curves are identical until roughly epoch 180 before diverging systematically.
- Note that the loss curves are not directly comparable across samplers, since each is averaged over a different sampling distribution; only the Recall@20 curves compare like with like.

### NGCF Reproduction

- Explain why NGCF required the same external verification as LightGCN: the Amazon results contain a negative claim about a published method, and such a claim is only as strong as the implementation behind it.
- Report reproduction to within 2.2% on Recall@20 using the same graph, evaluator and training loop as the LightGCN reproduction, so any difference between the two is attributable to the model rather than the pipeline.
- State the metric asymmetry honestly: recall reproduces closely, NDCG less so, most plausibly because this implementation uses a single message dropout where NGCF uses both message and node dropout.
- Note that both figures are lower bounds, since the selected regularisation sat at the top of the searched range and the run had not converged at the epoch limit.

### Baseline Audit and Density Reference

- Explain why matrix factorisation was also trained on Gowalla: the Amazon results were being explained by appeal to data density, and that explanation is an inference unless the same models are compared on data of a different density using the same code.
- Report that matrix factorisation was given its own regularisation sweep rather than inheriting LightGCN's, since on the Amazon data its optimum was several hundred times larger.
- Report the density result and state plainly that it contradicts the sparsity explanation: LightGCN's advantage over matrix factorisation is smaller on the denser dataset, not larger. Redirect the explanation to temporal structure in the evaluation, and carry this into Notebook 10's discussion.
- Report the baseline audit as a finding in its own right: a properly tuned BPR-MF exceeds the published MF baseline on the same files by 20%, which reduces NGCF's published advantage over matrix factorisation from 11.68% to 1.1%, and LightGCN's from 41% to 17.5%.
- Attribute the gain to the two things already documented in this project - user-uniform sampling, worth about 4.6% on Recall@20 here, and a regularisation sweep with an interior optimum - rather than to anything novel.
- Connect this to Rendle et al. and Ferrari Dacrema et al. on under-tuned baselines in recommender research, and note that this is a direct measurement on the benchmark's own data using code verified to reproduce both papers, rather than a citation.
- State the caveats: best-epoch selection here against validation-based early stopping in the original, so this implementation gets the more generous estimator; NGCF re-implemented NDCG, so Recall@20 is the safer basis for comparison; and the Gowalla NGCF and MF arms are single-seed.

### Discussion

- Discuss what the sampler result implies for reproducibility in this literature: an implementation detail absent from the paper was worth more than the architectural change the paper argues for.
- Explain why user-uniform sampling suits a per-user-averaged metric, and why the justification is fidelity and consistency rather than performance.
- Discuss the residual 0.25% on recall and the contribution of maximum-over-evaluations reporting.
- Draw the conclusion the NGCF reproduction enables: the same implementation reaches 98% of published recall on Gowalla and loses to matrix factorisation on Amazon Electronics, so the difference is attributable to the data regime rather than the code.
- State the limitation that GraphSAGE cannot be verified this way. It is not a standard collaborative filtering baseline with published figures on these datasets, and its output normalisation was disabled after testing. It is an adaptation rather than a reproduction and should be described as such throughout.
- Note that a seed standard deviation of 4e-5 means differences between models in the following chapter are easily resolvable.


### Key Quantitative Values

**Gowalla (LightGCN's released split):**
- 29,858 users, 40,981 items, 1,027,370 interactions, density 0.00084
- train 810,128 / test 217,242; median 4 test items per user

**LightGCN reproduction - 3 layers, dimension 64, batch 2048, 1000 epochs, 3 seeds:**
- Recall@20 0.18184 ± 0.00004 (published 0.1823, −0.25%)
- NDCG@20 0.15542 ± 0.00041 (published 0.1555, within seed variance)

**NGCF reproduction - 3 layers, dimension 64, batch 1024, 400 epochs, 1 seed:**
- Recall@20 0.15349 (published 0.1569, −2.2%)
- NDCG@20 0.12224 (published 0.1327, −7.9%)
- selected regularisation 1e-3; at 1e-5 the model peaks at epoch 25 (Recall@20 0.12847) and then overfits

**Sampling scheme, seed 42:**
- interaction-uniform: Recall@20 0.1738, NDCG@20 0.1482 (−4.6%, −4.7%)
- user-uniform: Recall@20 0.1819, NDCG@20 0.1551 (−0.22%, −0.26%)
- sampler effect 4.6% on Recall@20 against 3.9% for one layer to three in the paper's Table 3

**Implementation checks:**
- 0-layer identity: exact at initialisation, 1e-7 after five epochs
- propagation vs hand computation: 7.45e-09
- LightGCN parameters at dimension 32: 11,644,640, exactly the embedding tables
- NGCF +6,336 parameters, GraphSAGE +4,160

**Performance:**
- CSR adjacency halved epoch time: 7.46s → 3.77s
- 63 minutes per 1000-epoch LightGCN run; NGCF roughly 15s per epoch against LightGCN's 3.8s on the same graph
- RTX 5070 Ti Laptop GPU

**Gowalla baseline audit and density reference - Recall@20, identical configuration across arms:**
- BPR-MF (this project) 0.15515 at reg 1e-3; MF as published by NGCF 0.1291 (+20.2%)
- NGCF 0.15349 here, 0.1569 published; LightGCN 0.18184 here, 0.1823 published
- published NGCF over this baseline: +1.1% (its paper claims +11.68% over its own MF)
- published LightGCN over this baseline: +17.5% (implied +41% against the published MF)
- LightGCN over BPR-MF, both this project: +17.2%
- Gowalla 27 training interactions per user against Amazon Electronics' 7

# Notebook 10 - Collaborative Comparison

### Experimental Design

- State the controlled comparison explicitly: LightGCN, NGCF and GraphSAGE differ from the BPR-MF baseline in one respect only, how embeddings are propagated. Same loss, sampler, optimiser, dimension, evaluator and splits throughout.
- Explain that BPR-MF is not merely a baseline but the depth-0 member of the same family, verified exactly in Notebook 09, which is what makes the propagation-depth curve meaningful.
- Note that Notebook 10 supersedes Notebook 08's single-seed BPR-MF figure, since all four models are treated identically here.

### Tuning Methodology

- Describe coordinate descent and justify it against a full joint grid on cost grounds, stating the interaction it cannot capture as a limitation.
- Explain why hyperparameters were tuned separately for each protocol rather than transferred, referring back to Notebook 08 where the two protocols selected different regularisation for the same model.
- Report that graph models require regularisation one to two orders of magnitude weaker than matrix factorisation, and explain the mechanism: propagation is itself a regulariser, and the loss gradient reaching a layer-0 embedding is diluted through the propagation layers while the L2 penalty applies directly.
- Present the fixed-epoch-budget trap as a methodological finding: comparing hyperparameters at a fixed budget confounds regularisation strength with convergence speed, because stronger regularisation converges more slowly but reaches higher.
- Present the checkpoint-granularity finding for the same reason, and state plainly that correcting it moved the headline result against the hypothesis being tested.
- Justify the uniform 300-epoch budget from the convergence probe, and state the limitation that several timestamp configurations had not converged at that limit - equally, for every model.

### Implementation Issues

- Report that GraphSAGE could not train in its published form, and give the structural reason rather than treating it as a tuning failure: ReLU on the final layer plus L2 normalisation bounds every score to [0, 1], capping the margin BPR can open.
- Report that removing the final nonlinearity made GraphSAGE worse, and relate it correctly to LightGCN's ablation: their Table 1 shows removing nonlinearity alone also slightly reduced NGCF's performance, so this is a stronger instance of an effect they observed rather than counter-evidence. Note that the direct test of their claim - removing NGCF's own nonlinearity on this data - was not run, and belongs in future work.
- State that GraphSAGE is therefore an adaptation rather than a reproduction, and that this asymmetry with LightGCN and NGCF is acknowledged rather than hidden.

### Results

- Present the full comparison across both protocols, including Popularity and ItemKNN from Notebook 08 for context.
- Report the propagation depth curve with regularisation tuned at each depth, and show the fixed-regularisation curve alongside it to demonstrate why the joint sweep was necessary.
- Present the convergence figure, noting that it is measured on validation rather than test.
- Report that the metric ordering is identical under NDCG@10, HitRate@10 and Recall@20, so no conclusion depends on the metric.

### Discussion

- Lead with the protocol dependence: whether graph convolution helps on this data depends on which evaluation protocol is used, and the two protocols give opposite answers. Connect this to Ji et al. (2023) and to the Notebook 08 baseline reversal, which this extends to the graph models.
- Present the three-point pattern as the strongest form of the argument. LightGCN beats matrix factorisation by 17.2% on Gowalla (random split, no temporal shift), by 26.7% on Amazon under leave-last-out (training graph contains activity postdating many test items), and loses by a factor of 2.9 under the timestamp protocol (genuine temporal gap, no leakage). Density moves in the opposite direction to the effect, so the ordering is explained by temporal structure in the evaluation rather than by data sparsity.
- Advance, as a hypothesis rather than a conclusion, that graph propagation benefits disproportionately from temporal leakage: propagation aggregates a neighbourhood that encodes what was co-consumed during the training window, and where that window overlaps the test period the neighbourhood carries information about the future it should not have. Note that the clean test - Gowalla under a temporal split - is not possible with the released files.
- Discuss why LightGCN loses to ItemKNN under the favourable protocol, and what that implies about the strength of the baselines the GNN literature typically reports against.
- Tie the ItemKNN result to the Gowalla baseline audit in Notebook 09. Both point the same way: LightGCN's advantage depends heavily on what it is measured against, losing to a well-tuned neighbourhood method on Amazon and shrinking from 41% to 17.5% on Gowalla once the matrix factorisation baseline is tuned with the same care.
- Explain the saturation of propagation depth in terms of sparsity: with about seven interactions per user there is little informative signal two or three hops away.
- Argue that NGCF and GraphSAGE underperforming matrix factorisation supports LightGCN's case about unnecessary machinery, and note that this claim is credible here only because the same NGCF implementation reproduces its published result to within 2.2%.
- Discuss the seed variance difference between LightGCN and BPR-MF, and whether propagation should be understood as stabilising the solution as well as improving it.
- Acknowledge the limitations: single-seed tuning, coordinate descent rather than joint search, no depth sweep for NGCF or GraphSAGE under the timestamp protocol, and unconverged timestamp configurations.

### Key Quantitative Values

- 50 tuning runs, 28 final runs (three graph models across three seeds, BPR-MF across five, both protocols)
- Fixed budget 300 epochs, dimension 32, Adam 1e-3, batch 8192, user-uniform sampling
- Tuning on a 10,000-user validation subset; reported results on the full 50,000 evaluation users

**Selected configurations (layers, L2):**
- leave-last-out: LightGCN (3, 3e-3), NGCF (3, 1e-2), GraphSAGE (3, 3e-3), BPR-MF (0, 0.10)
- timestamp: LightGCN (3, 3e-4), NGCF (3, 1e-2), GraphSAGE (3, 3e-4), BPR-MF (0, 0.15)

**Test NDCG@10, leave-last-out:**
- ItemKNN 0.00948 > LightGCN 0.00812 ± 0.00007 > Popularity 0.00752 > BPR-MF 0.00641 ± 0.00069 > NGCF 0.00425 ± 0.00013 = GraphSAGE 0.00425 ± 0.00020
- LightGCN over BPR-MF: +27%

**Test NDCG@10, timestamp:**
- BPR-MF 0.00465 ± 0.00040 > ItemKNN 0.00217 > LightGCN 0.00158 ± 0.00020 > NGCF 0.00150 ± 0.00010 > GraphSAGE 0.00134 ± 0.00014 > Popularity 0.00124
- BPR-MF over LightGCN: 2.9x

**Propagation depth (validation NDCG@10, leave-last-out, regularisation tuned per depth):**
- 0 layers (BPR-MF) 0.00953, 1 layer 0.012128, 2 layers 0.012128, 3 layers 0.012740, 4 layers 0.012308
- gains over BPR-MF: +27.3%, +27.3%, +33.7%, +29.2%
- fixed-regularisation sweep understates 1 layer by 11.6%

**GraphSAGE output normalisation:**
- validation NDCG@10 0.000976 with normalisation, 0.006032 without (6x)
- removing the final ReLU: 0.004750 against 0.006393 (-26%)

**Optimal regularisation by depth (LightGCN, leave-last-out):** 1e-2, 3e-3, 3e-3, 1e-3 for 1 to 4 layers

**Seed variance:** LightGCN 0.9% of mean, BPR-MF 14%

**Checkpoint granularity:** evaluating BPR-MF every 10 epochs rather than every 2 cost 30% on test and would have overstated LightGCN's advantage as 67% rather than 27%

**Cost per epoch:** LightGCN 7.07s, NGCF 11.42s, GraphSAGE 6.87s

# Notebook 11 - Metadata and Cold-Start Recommendation

### Experimental Design

- State the three questions the notebook answers: whether metadata improves recommendation, whether the choice of encoder matters, and whether metadata helps where there is no interaction history.
- Explain why only LightGCN was carried forward, and why matrix factorisation was added as a control rather than as a competitor: the interesting question is not whether content helps, but whether it helps *because of* the graph.
- Explain the zero-initialised feature projection and why it makes this a strict nesting. The two arms begin numerically identical, so the feature-free results from Notebook 10 can be reused directly and any difference is attributable to the metadata alone.
- Note that matrix factorisation with features was implemented as LightGCN with zero propagation layers, reusing code verified in Notebook 09 rather than adding a second feature hook.
- Justify giving each arm its own regularisation sweep, and quantify why it mattered: features moved LightGCN's optimum from 3e-3 to 1e-4, and inheriting the old value would have reported a 2% gain rather than 15%.

### Metadata Effect

- Present the four-way table: two models, two protocols, with and without features.
- Emphasise that the matrix factorisation comparisons use identical regularisation in both arms, so the negative result is not a tuning artefact.
- Report that the benefit to LightGCN is seven times larger under the timestamp protocol than under leave-last-out, and connect this to Notebook 10: features matter most where the collaborative signal is weakest.
- State clearly that metadata does not overturn the Notebook 10 conclusion. Matrix factorisation still wins the leakage-free protocol with or without it.
- Use the regularisation behaviour as supporting evidence for the mechanism: adding features moved LightGCN's optimum down thirtyfold, while matrix factorisation's optimum did not move, which is what you would expect if the projection is useful information in one case and unusable capacity in the other.

### Encoder Comparison

- Present this as a supporting result, appropriate for an appendix.
- Report the three encoders and state that they are indistinguishable, reasoning from the seed variance measured for BLaIR rather than from seeds run for each encoder. State that this is the basis for the claim.
- Draw the conclusion that matters: the benefit comes from having a text representation of the product at all, not from a sophisticated one. Domain-specific pre-training supplied with the dataset gave no measurable advantage over a general-purpose encoder or over TF-IDF with SVD.
- Note the limitation that the three feature matrices have different spreads, and that the L2 penalty applies to the projected features, so identical regularisation is not identical shrinkage across encoders.

### Cold-Start Evaluation

- Report the cold-item counts for both protocols first, and treat the difference as a finding rather than a setup detail: leave-last-out affects 0.7% of evaluation users against 44.3% under the timestamp protocol. This is a second respect in which the standard protocol understates the real task.
- Explain why the experiment was run under two framings, and that they answer different questions: whether a new product can appear in an unrestricted top-10 list, and whether new products can be ranked sensibly among themselves.
- Report the open-ranking result as a negative finding with a mechanism: a cold item's embedding is never updated during training, so it stays near its random initialisation, and warm items crowd it out. Conclude that surfacing new products requires a deliberate mechanism, such as reserved slots or a restricted candidate pool, rather than better item representations.
- Report the restricted-pool result as the positive finding, with the random baseline included so the small absolute values are interpretable.
- Present the inversion as the headline: the weakest model overall is the strongest on cold items, by a factor of two. This is the property GraphSAGE was included to test and the only setting in the project where it wins.

### Discussion

- Argue that content and propagation are complementary. Propagation gives a feature-derived representation somewhere to go, smoothing it across the graph neighbourhood where it reinforces genuine co-purchase structure; without propagation the same projection is capacity the model overfits.
- Discuss what the inversion implies for evaluation practice: aggregate metrics are dominated by warm items and conceal the behaviour that matters for new products. A system chosen on aggregate NDCG would select the model that performs worst where a retailer most needs help.
- Connect the cold-start item counts back to the protocol argument in Notebook 10. Leave-last-out understates both temporal leakage and cold-start difficulty, and these are independent problems with the same protocol.
- Note the validation-test divergence observed for matrix factorisation with features, and connect it to the checkpoint-selection problem in Notebook 10: under leave-last-out, validation is each user's second-to-last item and test is their last, so a weakly regularised model that memorises recent history scores well on one and badly on the other.
- State the limitations: the encoder comparison is single-seed; GraphSAGE with features was tuned only on leave-last-out and used untuned under the timestamp protocol; NGCF was not given features; and no content-based ItemKNN was built, which is the natural baseline this project lacks.

### Key Quantitative Values

- Features: BLaIR 807 dimensions, MiniLM 423, TF-IDF+SVD 295, all centred, no item with an empty vector
- Selected regularisation with features: LightGCN 1e-4 (leave-last-out) and 3e-5 (timestamp), against 3e-3 and 3e-4 without; matrix factorisation unchanged at 0.10 and 0.15

**Metadata effect, test NDCG@10:**
- LightGCN, leave-last-out: 0.00812 → 0.00864 (+6.3%)
- LightGCN, timestamp: 0.00158 → 0.00230 (+45.9%)
- BPR-MF, leave-last-out: 0.00630 → 0.00574 (−8.9%)
- BPR-MF, timestamp: 0.00454 → 0.00422 (−7.0%)
  (BPR-MF baselines restricted to the three seeds used here; Notebook 10 quotes 0.00641 and 0.00465 over five)
- GraphSAGE, leave-last-out: 0.00445 → 0.00537 (+20.7%, single seed)

**Encoders, leave-last-out, seed 42, against 0.00811 without features:**
- MiniLM 0.00873 (+7.7%), BLaIR 0.00872 (+7.6%), TF-IDF+SVD 0.00857 (+5.7%)
- range across encoders ≈ 1.5 standard deviations of BLaIR's three-seed variance (0.00011)

**Cold items:**
- leave-last-out: 425 items, 1,817 test interactions, 338 users (0.7% of evaluation users)
- timestamp: 8,353 items, 64,088 test interactions, 22,130 users (44.3%)

**Cold-item ranking, timestamp, restricted candidate pool, NDCG@10:**
- GraphSAGE + BLaIR 0.00232 (5.0× random)
- LightGCN + BLaIR 0.00116 (2.5×)
- LightGCN, no features 0.00086 (1.9×)
- BPR-MF, no features 0.00060 (1.3×)
- BPR-MF + BLaIR 0.00053 (1.2×)
- random 0.00046
- open ranking against the full catalogue: every model at or below chance (random 0.00004, best model 0.00003, six of eight exactly zero)
- value of features on cold items: GraphSAGE 5.92x, LightGCN 1.34x, BPR-MF 0.88x

**Verification:** retrained cold-start models reproduce Notebooks 10 and 11 to five decimal places, including the zero-layer identity for matrix factorisation


# Notebook 12 - Significance, Scaling and Synthesis

### Significance Testing

- Explain why per-user scores were saved from the outset: comparing models on means alone cannot distinguish a broad small improvement from a narrow large one.
- Justify the Wilcoxon signed-rank test over a paired t-test on the grounds that per-user NDCG is zero-inflated and non-normal, and justify pairing on the grounds that users differ enormously in predictability.
- Explain why the tests use a single seed while the reported means use all available seeds, and state which way that biases the comparison.
- Report the results, but lead with effect sizes rather than p-values, and explain why: with 50,000 paired users, statistical significance is close to guaranteed and carries little information.
- Report that roughly 97% of users are tied in every comparison, and draw the consequence: all reported differences are decided by a small minority of users. This belongs in the results chapter, not a footnote.
- Correct the Notebook 11 claim about metadata harming matrix factorisation: significant under leave-last-out, consistent in direction but not significant under timestamp.

### Scaling

- Present the cost measurement as a finding in its own right, not as setup for the accuracy comparison.
- Report the fitted exponent and explain the mechanism: every batch propagates over the entire graph, and the number of batches grows with the data, so cost grows with the square of dataset size.
- Report wall-clock ratios rather than per-epoch ratios, since LightGCN also needs more epochs and the per-epoch figure understates what a practitioner pays.
- State the feasibility result plainly: matrix factorisation trains on 12.2 million interactions in 39 minutes, LightGCN would need roughly 25 hours, and this is why the accuracy comparison stops at the medium subset.
- Note that memory was not the constraint, which is worth saying because it is the limitation readers usually assume.
- Explain why absolute scores fall with dataset size - the catalogue grows too - and therefore why the comparison is about ordering and gaps rather than levels.
- State the hyperparameter limitation honestly and quantify it using the partial re-tuning check, rather than naming it and moving on.

### Synthesis

- Present the two ranked tables side by side, or the slope chart, as the single clearest statement of the protocol argument.
- Draw attention to Popularity specifically: fourth under leave-last-out, last under timestamp, with nothing about the model changed.
- Note that metadata lifts LightGCN one place under each protocol without changing which model wins either.

### Discussion

- Argue that the choice of evaluation protocol determines the answer to the dissertation's research question, and that this is the central result rather than a caveat.
- Connect the cost finding to the accuracy finding: LightGCN's advantage under leave-last-out comes at 22 times the training cost, and that multiple grows with data size. Whether the trade is worth making is a deployment question, and the answer at realistic catalogue sizes is not obviously yes.
- Discuss the 97% tie rate and what it implies about the practical value of any of these models at full-catalogue ranking.
- State the limitations: scaling measured on one protocol, hyperparameters not re-tuned per size, single seed for the scaling runs, and LightGCN not trainable at full scale within the compute budget.

### Key Quantitative Values

**Significance, paired Wilcoxon on per-user NDCG@10, seed 42:**
- LightGCN over BPR-MF, leave-last-out: +17.0%, p = 5.8e-05
- BPR-MF over LightGCN, timestamp: +178.0%, p = 2.2e-41
- ItemKNN over LightGCN, leave-last-out: +17.0%, p = 2.4e-04
- LightGCN+BLaIR over LightGCN: +7.6% (p = 1.8e-03) and +32.4% (p = 1.5e-05)
- BPR-MF+BLaIR against BPR-MF: −14.2% (p = 2.0e-04) and −8.1% (p = 0.78)
- approximately 97% of users tied in every comparison

**Scaling, leave-last-out, seed 42:**
- subsets: 1.83M / 4.95M / 12.19M training interactions
- LightGCN seconds per epoch: 6.88 / 49.98 / 303.41; fitted exponent 1.996
- BPR-MF seconds per epoch: 0.84 / 4.65 / 19.00
- wall-clock ratio LightGCN to BPR-MF: 13.8x on small, 22.1x on medium
- peak GPU memory on full: 5.4 GB
- NDCG@10 on small / medium: ItemKNN 0.00948 / 0.00878, LightGCN 0.00811 / 0.00705, BPR-MF 0.00693 / 0.00447
- BPR-MF on full: 0.00209 in 39 minutes
- BPR-MF regularisation optimum on medium: 0.03 (0.00471) against 0.10 selected on small (0.00447)

**Synthesis, rank changes between protocols:**
- BPR-MF 5th to 1st, Popularity 4th to 8th, ItemKNN 1st to 4th