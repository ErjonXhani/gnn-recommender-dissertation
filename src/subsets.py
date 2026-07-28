"""Utilities for creating nested interaction subsets.

These functions are used in Notebook 05 to build the small, medium and full
datasets used in the scalability experiments.
"""

from __future__ import annotations

from typing import Dict, Mapping, Tuple

import numpy as np
import pandas as pd

__all__ = [
    "nested_user_samples",
    "iterative_k_core",
    "reencode",
    "subset_statistics",
]


def nested_user_samples(
    user_ids: np.ndarray,
    fractions: Mapping[str, float],
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Generate nested user samples from a single random permutation."""
    users = np.unique(user_ids)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(users)

    samples = {}
    for name, fraction in sorted(fractions.items(), key=lambda kv: kv[1]):
        if not 0 < fraction <= 1:
            raise ValueError(f"fraction for '{name}' must be in (0, 1], got {fraction}")
        n = len(perm) if fraction >= 1.0 else int(round(fraction * len(perm)))
        samples[name] = np.sort(perm[:n])
    return samples


def iterative_k_core(
    interactions: pd.DataFrame,
    k: int = 5,
    max_iterations: int = 100,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Repeatedly drop users and items with fewer than k interactions.

    One pass is not enough: removing a low-degree user can push an item below the
    threshold, which can then push other users below it. The loop runs until a pass
    removes nothing.

    Returns the filtered frame and a small dict describing what happened.
    """
    df = interactions
    history = [len(df)]

    for iteration in range(1, max_iterations + 1):
        users = df["user_idx"].to_numpy()
        items = df["item_idx"].to_numpy()

        # bincount is much faster than groupby here, and the indices are already
        # non-negative integers so it works directly
        user_counts = np.bincount(users)
        item_counts = np.bincount(items)
        keep = (user_counts[users] >= k) & (item_counts[items] >= k)

        if keep.all():
            return df.reset_index(drop=True), {
                "iterations": iteration,
                "rows_in": history[0],
                "rows_out": len(df),
                "history": history,
            }

        df = df[keep]
        history.append(len(df))

    raise RuntimeError(
        f"k-core did not converge in {max_iterations} iterations "
        f"(last sizes: {history[-5:]})"
    )


def reencode(
    interactions: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Renumber users and items to contiguous 0..N-1 indices.

    Embedding layers expect contiguous indices. After subsetting, the original
    indices contain gaps, so users and items are re-encoded while keeping mappings
    back to the global IDs.

    Returns (reindexed interactions, user mapping, item mapping).
    """
    global_users = np.unique(interactions["user_idx"].to_numpy())
    global_items = np.unique(interactions["item_idx"].to_numpy())

    # both arrays are sorted, so searchsorted gives the new index directly
    local_users = np.searchsorted(global_users, interactions["user_idx"].to_numpy())
    local_items = np.searchsorted(global_items, interactions["item_idx"].to_numpy())

    out = interactions.copy()
    out["user_idx"] = local_users.astype("int32")
    out["item_idx"] = local_items.astype("int32")

    user_mapping = pd.DataFrame({
        "user_idx": np.arange(len(global_users), dtype="int32"),
        "global_user_idx": global_users.astype("int32"),
    })
    item_mapping = pd.DataFrame({
        "item_idx": np.arange(len(global_items), dtype="int32"),
        "global_item_idx": global_items.astype("int32"),
    })

    return out.reset_index(drop=True), user_mapping, item_mapping


def subset_statistics(interactions: pd.DataFrame) -> Dict[str, float]:
    """Size and sparsity figures for one subset."""
    n_interactions = len(interactions)
    n_users = interactions["user_idx"].nunique()
    n_items = interactions["item_idx"].nunique()

    return {
        "interactions": int(n_interactions),
        "users": int(n_users),
        "items": int(n_items),
        "density": float(n_interactions / (n_users * n_items)) if n_users and n_items else 0.0,
        "mean_user_degree": float(n_interactions / n_users) if n_users else 0.0,
        "mean_item_degree": float(n_interactions / n_items) if n_items else 0.0,
    }