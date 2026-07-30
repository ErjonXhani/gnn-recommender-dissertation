"""Non-GNN baseline recommenders.

Each exposes a ``score_fn`` compatible with ``evaluation.Evaluator``: given a batch of
user ids, return a (batch, n_items) score matrix.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
import torch

__all__ = [
    "PopularityRecommender",
    "ItemKNNRecommender",
    "random_score_fn",
]


class PopularityRecommender:
    """Rank items by how often they appear in training, identically for every user.

    No personalisation at all. It serves as a minimum baseline that more complex 
    models should outperform and is simple enough to validate
    independently of the evaluation pipeline.
    """

    def __init__(self):
        self.counts: np.ndarray | None = None

    def fit(self, train: pd.DataFrame, n_items: int) -> "PopularityRecommender":
        self.counts = np.bincount(
            train["item_idx"].to_numpy(), minlength=n_items
        ).astype(np.float32)
        return self

    def score_fn(self, device: str = "cuda") -> Callable[[torch.Tensor], torch.Tensor]:
        if self.counts is None:
            raise RuntimeError("call fit() first")
        scores = torch.as_tensor(self.counts, device=device)
        return lambda users: scores.expand(len(users), -1).clone()


def random_score_fn(n_items: int, device: str = "cuda", seed: int = 0):
    """Uniform random scores. Used to check the evaluator against a known expectation."""
    generator = torch.Generator(device=device).manual_seed(seed)
    return lambda users: torch.rand(
        len(users), n_items, generator=generator, device=device
    )


class ItemKNNRecommender:
    """Item-item collaborative filtering with shrunk cosine similarity.

    User-item scores are computed by summing each candidate item's similarity to the
    items in the user's training history.

    Shrinkage reduces unreliable high similarities caused by very low co-occurrence
    counts in sparse data. Because the complete item-item matrix is impractical,
    only the top ``n_neighbours`` similarities per item are retained.
    """

    def __init__(self, n_neighbours: int = 100, shrinkage: float = 0.0):
        self.n_neighbours = n_neighbours
        self.shrinkage = shrinkage
        self.similarity = None   # scipy sparse, item x item
        self.user_items = None   # scipy sparse, user x item

    def fit(self, train: pd.DataFrame, n_users: int, n_items: int,
            chunk_size: int = 2000, verbose: bool = True) -> "ItemKNNRecommender":
        from scipy import sparse

        interactions = sparse.csr_matrix(
            (np.ones(len(train), dtype=np.float32),
             (train["user_idx"].to_numpy(), train["item_idx"].to_numpy())),
            shape=(n_users, n_items),
        )
        self.user_items = interactions

        counts = np.asarray(interactions.sum(axis=0)).ravel().astype(np.float32)
        interactions_csc = interactions.tocsc()

        blocks = []
        for start in range(0, n_items, chunk_size):
            end = min(start + chunk_size, n_items)

            block = (interactions_csc[:, start:end].T @ interactions).tocoo()

            denominator = np.sqrt(
                counts[block.row + start] * counts[block.col]
            ) + self.shrinkage
            block.data = block.data / np.clip(denominator, 1e-12, None)

            blocks.append(self._keep_top_k(block.tocsr(), self.n_neighbours, offset=start))

            if verbose and (start // chunk_size) % 10 == 0:
                print(f"  items {end:,}/{n_items:,}", flush=True)

        self.similarity = sparse.vstack(blocks).tocsr()
        return self

    @staticmethod
    def _keep_top_k(block, k: int, offset: int = 0):
        """Keep the k largest values per row, excluding the item itself.

        ``offset`` is the global index of the block's first row, so row i is item
        ``offset + i``. An item is trivially its own nearest neighbour and would
        otherwise take one of the k slots for nothing.
        """
        from scipy import sparse

        data, indices, indptr = block.data, block.indices, block.indptr
        keep_data, keep_indices, keep_indptr = [], [], [0]

        for row in range(block.shape[0]):
            start, end = indptr[row], indptr[row + 1]
            row_data, row_indices = data[start:end], indices[start:end]

            not_self = row_indices != (offset + row)
            row_data, row_indices = row_data[not_self], row_indices[not_self]

            if len(row_data) > k:
                top = np.argpartition(row_data, -k)[-k:]
                row_data, row_indices = row_data[top], row_indices[top]

            keep_data.append(row_data)
            keep_indices.append(row_indices)
            keep_indptr.append(keep_indptr[-1] + len(row_data))

        return sparse.csr_matrix(
            (np.concatenate(keep_data), np.concatenate(keep_indices),
             np.array(keep_indptr)),
            shape=block.shape,
        )

    def score_fn(self, device: str = "cuda"):
        if self.similarity is None:
            raise RuntimeError("call fit() first")

        def scores(users: torch.Tensor) -> torch.Tensor:
            user_rows = self.user_items[users.cpu().numpy()]
            dense = np.asarray((user_rows @ self.similarity).todense(), dtype=np.float32)
            return torch.as_tensor(dense, device=device)

        return scores