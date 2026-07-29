"""Build item features from the cleaned product metadata.

Prepares metadata for encoding and removes fields that would introduce
temporal leakage.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import pandas as pd

import torch

from pathlib import Path
import pyarrow.parquet as pq

__all__ = [
    "strip_leaky_fields",
    "strip_leaky_from_combined",
    "text_length_stats",
    "encode_blair",
    "build_tfidf_svd",
    "assemble_features",
    "slice_for_subset",
    "load_item_features",
]


def strip_leaky_fields(
    text: pd.Series,
    keys: Sequence[str],
    separator: str = " | ",
) -> pd.Series:
    """Remove separator-delimited metadata fields matching the given keys.

    Used for ``details_text``, where each metadata entry begins with its key.
    """
    prefixes = tuple(keys)

    def clean(value: str) -> str:
        if not isinstance(value, str) or not value:
            return value
        parts = [p for p in value.split(separator) if not p.strip().startswith(prefixes)]
        return separator.join(parts)

    return text.map(clean)


def strip_leaky_from_combined(
    combined: pd.Series,
    details: pd.Series,
    keys: Sequence[str],
    separator: str = " | ",
) -> tuple[pd.Series, int]:
    """Remove post-cutoff metadata from the combined text.

    Stripping cannot be done directly on the combined text: the first details key is
    joined to the end of the description with a space rather than a pipe, so it does
    not start its own segment. Instead the details field is cleaned on its own, where
    the delimiting is reliable, and the result substituted back in.

    Returns the cleaned text and the number of rows where the details string could not
    be located in the combined text (should be zero).
    """
    clean_details = strip_leaky_fields(details, keys, separator)

    out, missing = [], 0
    for full, original, cleaned in zip(combined, details, clean_details):
        if not isinstance(full, str):
            out.append(full)
            continue
        if isinstance(original, str) and original and original in full:
            out.append(full.replace(original, cleaned, 1))
        else:
            if isinstance(original, str) and original:
                missing += 1
            out.append(full)

    return pd.Series(out, index=combined.index), missing





def text_length_stats(text: pd.Series) -> Dict[str, float]:
    """Character and whitespace-token lengths, to judge truncation."""
    chars = text.fillna("").str.len()
    words = text.fillna("").str.split().str.len().fillna(0)
    return {
        "mean_chars": float(chars.mean()),
        "median_chars": float(chars.median()),
        "p95_chars": float(chars.quantile(0.95)),
        "mean_words": float(words.mean()),
        "median_words": float(words.median()),
        "p95_words": float(words.quantile(0.95)),
        "max_words": float(words.max()),
    }

@torch.no_grad()
def encode_blair(
    texts: Sequence[str],
    model,
    tokenizer,
    device: str = "cuda",
    batch_size: int = 64,
    max_tokens: int = 512,
    use_fp16: bool = True,
    progress_every: int = 200,
) -> np.ndarray:
    """Encode texts with BLaIR following the released model usage.

    Uses CLS pooling with L2 normalisation rather than mean pooling, matching the
    model's intended embedding space.

    Texts are length-sorted before batching to reduce unnecessary padding while
    preserving the original output order.
    """
    model = model.to(device).eval()
    order = np.argsort([len(t) for t in texts])
    out = np.zeros((len(texts), model.config.hidden_size), dtype=np.float32)

    n_batches = (len(order) + batch_size - 1) // batch_size
    for batch_i, start in enumerate(range(0, len(order), batch_size)):
        idx = order[start:start + batch_size]
        encoded = tokenizer(
            [texts[i] for i in idx],
            padding=True,
            truncation=True,
            max_length=max_tokens,
            return_tensors="pt",
        ).to(device)

        with torch.autocast(
            device_type="cuda", dtype=torch.float16,
            enabled=use_fp16 and device == "cuda",
        ):
            hidden = model(**encoded, return_dict=True).last_hidden_state

        pooled = hidden[:, 0].float()
        pooled = pooled / pooled.norm(dim=1, keepdim=True).clamp(min=1e-12)
        out[idx] = pooled.cpu().numpy()

        if progress_every and (batch_i + 1) % progress_every == 0:
            print(f"  {batch_i + 1}/{n_batches} batches", flush=True)

    return out


def build_tfidf_svd(
    texts: Sequence[str],
    n_components: int = 256,
    max_features: int = 50_000,
    min_df: int = 5,
    seed: int = 42,
) -> np.ndarray:
    """Bag-of-words baseline: TF-IDF reduced by truncated SVD (LSA).

    Output is L2-normalised so it sits on the same scale as the neural encoders,
    which both produce unit vectors.
    """
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectoriser = TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        stop_words="english",
        sublinear_tf=True,
        dtype=np.float32,
    )
    sparse = vectoriser.fit_transform(texts)

    svd = TruncatedSVD(n_components=n_components, algorithm="randomized",
                       random_state=seed)
    reduced = svd.fit_transform(sparse).astype(np.float32)

    norms = np.linalg.norm(reduced, axis=1, keepdims=True)
    return reduced / np.clip(norms, 1e-12, None), float(svd.explained_variance_ratio_.sum())


def assemble_features(
    text: np.ndarray,
    categorical: np.ndarray,
    centre: bool = True,
) -> np.ndarray:
    """Combine text embeddings with categorical features.

    Text embeddings are centred and re-normalised before concatenation to improve
    their conditioning. The model's projection layer learns the relative weighting
    between the text and categorical blocks.
    """
    block = text.astype(np.float32, copy=True)
    if centre:
        block -= block.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(block, axis=1, keepdims=True)
    block /= np.clip(norms, 1e-12, None)

    return np.concatenate([block, categorical.astype(np.float32)], axis=1)


def slice_for_subset(features: np.ndarray, item_mapping: pd.DataFrame) -> np.ndarray:
    """Take the rows of a global feature matrix belonging to one subset.

    ``item_mapping`` runs 0..N-1 in local index order with the global index alongside,
    so indexing by the global column returns rows already in local order.
    """
    global_idx = item_mapping.sort_values("item_idx")["global_item_idx"].to_numpy()
    return features[global_idx]


ENCODER_FILES = {
    "blair": "blair_embeddings.npy",
    "minilm": "minilm_embeddings.npy",
    "tfidf_svd": "tfidf_svd_embeddings.npy",
}


def load_item_features(
    features_dir,
    subsets_dir,
    subset: str,
    encoder: str = "blair",
    centre: bool = True,
) -> np.ndarray:
    """Load the item feature matrix for one subset.

    Features are assembled from the saved global embeddings and then sliced into
    the subset's local item order.
    """
    if encoder not in ENCODER_FILES:
        raise ValueError(f"unknown encoder '{encoder}', expected one of {list(ENCODER_FILES)}")

    features_dir, subsets_dir = Path(features_dir), Path(subsets_dir)
    text = np.load(features_dir / ENCODER_FILES[encoder])
    categorical = np.load(features_dir / "category_onehot.npy")

    combined = assemble_features(text, categorical, centre=centre)
    mapping = pq.read_table(subsets_dir / subset / "item_mapping.parquet").to_pandas()
    return slice_for_subset(combined, mapping)