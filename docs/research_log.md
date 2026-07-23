## 2026-07-23

### Completed
- Implemented streaming data processing for the Amazon Reviews dataset.
- Added automatic caching of dataset statistics.
- Validated the cache loading mechanism.
- Generated summary statistics and exploratory visualisations.

### Design Decisions
- Selected `parent_asin` as the product identifier because it represents the parent product across variants.
- Used `Counter` objects instead of loading the full dataset into memory to reduce RAM usage.
- Cached computed statistics to avoid repeated full-dataset scans.

### Next Steps
- Implement the preprocessing pipeline.
- Filter inactive users and products.
- Prepare temporal train/validation/test splits.