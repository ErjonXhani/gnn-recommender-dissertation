"""Recommendation models.

Every model exposes the same interface: ``forward()`` returns the final user and item
embedding tables, and scoring is their dot product. Matrix factorisation returns its
embeddings unchanged, while graph-based models return them after message passing.
Keeping the interface identical allows every model to use the same training loop and
evaluation pipeline.
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["MatrixFactorization"]


class MatrixFactorization(nn.Module):
    """Free user and item embeddings scored by dot product.

    Trained with BPR this is BPR-MF. It is also LightGCN with zero propagation
    layers, which makes it the controlled comparison for the graph models: same loss,
    same sampler, same dimension, differing only in whether propagation happens.
    """

    def __init__(self, n_users: int, n_items: int, dim: int = 64, seed: int = 42):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)

        self.n_users, self.n_items, self.dim = n_users, n_items, dim
        self.user_embedding = nn.Parameter(
            torch.randn(n_users, dim, generator=generator) * 0.1)
        self.item_embedding = nn.Parameter(
            torch.randn(n_items, dim, generator=generator) * 0.1)

    def forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.user_embedding, self.item_embedding

    @torch.no_grad()
    def score_fn(self):
        """Scoring function for the evaluator."""
        def scores(users: torch.Tensor) -> torch.Tensor:
            user_emb, item_emb = self.forward()
            return user_emb[users] @ item_emb.T
        return scores