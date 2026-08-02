"""Recommendation models.

Every model exposes the same interface: ``forward()`` returns the final user and item
embedding tables, and scoring is their dot product. Matrix factorisation returns its
embeddings unchanged, while graph-based models return them after message passing.
Keeping the interface identical allows every model to use the same training loop and
evaluation pipeline.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "MatrixFactorization",
    "GraphRecommender",
    "LightGCN",
    "NGCF",
    "GraphSAGE",
]


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

    def base_embeddings(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Same as forward. MF has no propagation, so layer 0 is also the final layer."""
        return self.user_embedding, self.item_embedding

    @torch.no_grad()
    def score_fn(self):
        """Scoring function for the evaluator."""
        def scores(users: torch.Tensor) -> torch.Tensor:
            user_emb, item_emb = self.forward()
            return user_emb[users] @ item_emb.T
        return scores



class GraphRecommender(nn.Module):
    """Base class for the graph models.

    Every model here builds its layer-0 embeddings identically and differs only in how
    it propagates them. That is what makes the comparison attributable to propagation:
    the embedding tables, the optional feature projection, the scoring function and the
    training loop are all shared.

    The adjacency must already be on the target device when the model is constructed;
    sparse tensors are not moved by ``.to()``.
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        dim: int = 64,
        item_features: np.ndarray | None = None,
        seed: int = 42,
    ):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)

        self.n_users, self.n_items, self.dim = n_users, n_items, dim
        self.user_embedding = nn.Parameter(
            torch.randn(n_users, dim, generator=generator) * 0.1)
        self.item_embedding = nn.Parameter(
            torch.randn(n_items, dim, generator=generator) * 0.1)

        self.use_features = item_features is not None
        if self.use_features:
            self.register_buffer(
                "item_features",
                torch.as_tensor(item_features, dtype=torch.float32),
            )
            self.feature_projection = nn.Linear(item_features.shape[1], dim, bias=False)
            # zero init means the model starts exactly equal to its feature-free
            # counterpart, so the metadata ablation is a strict nesting rather than a
            # comparison of two different models. Gradients still reach W through the
            # item embeddings, so it trains normally.
            nn.init.zeros_(self.feature_projection.weight)

    def base_embeddings(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Layer-0 embeddings, identical across all models."""
        items = self.item_embedding
        if self.use_features:
            items = items + self.feature_projection(self.item_features)
        return self.user_embedding, items

    def _split(self, nodes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return nodes[: self.n_users], nodes[self.n_users :]

    def propagate(self, nodes: torch.Tensor):
        raise NotImplementedError

    def forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        users, items = self.base_embeddings()
        return self.propagate(torch.cat([users, items], dim=0))

    @torch.no_grad()
    def score_fn(self):
        def scores(users: torch.Tensor) -> torch.Tensor:
            user_emb, item_emb = self.forward()
            return user_emb[users] @ item_emb.T
        return scores


class LightGCN(GraphRecommender):
    """He et al. (2020). Neighbourhood aggregation with nothing else.

    No feature transformation, no non-linearity, no self-loops. The final
    representation is the mean of every layer including layer 0, which is the uniform
    layer weighting the paper settles on.

    With ``n_layers=0`` this reduces exactly to matrix factorisation, which is the
    identity used to verify the implementation.
    """

    def __init__(self, *args, adjacency: torch.Tensor, n_layers: int = 3, **kwargs):
        super().__init__(*args, **kwargs)
        self.adjacency = adjacency
        self.n_layers = n_layers

    def propagate(self, nodes: torch.Tensor):
        layers = [nodes]
        for _ in range(self.n_layers):
            nodes = torch.sparse.mm(self.adjacency, nodes)
            layers.append(nodes)
        return self._split(torch.stack(layers, dim=0).mean(dim=0))


class NGCF(GraphRecommender):
    """Wang et al. (2019). Adds feature transformation and element-wise interaction.

    Each layer combines a transformed self-and-neighbour term with a term capturing
    the element-wise product between a node and its neighbours, then applies LeakyReLU.
    Layer outputs are concatenated rather than averaged, so the final representation is
    ``dim x (n_layers + 1)`` wide.

    This is the model LightGCN argued was over-complicated, so it is the direct
    comparison for whether that added machinery helps here.
    """

    def __init__(self, *args, adjacency: torch.Tensor, n_layers: int = 3,
                 dropout: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.adjacency = adjacency
        self.n_layers = n_layers
        self.dropout = dropout

        self.self_transform = nn.ModuleList(
            [nn.Linear(self.dim, self.dim) for _ in range(n_layers)])
        self.interaction_transform = nn.ModuleList(
            [nn.Linear(self.dim, self.dim) for _ in range(n_layers)])

    def propagate(self, nodes: torch.Tensor):
        layers = [nodes]
        for layer in range(self.n_layers):
            neighbours = torch.sparse.mm(self.adjacency, nodes)
            message = (
                self.self_transform[layer](nodes + neighbours)
                + self.interaction_transform[layer](neighbours * nodes)
            )
            nodes = F.leaky_relu(message, negative_slope=0.2)
            nodes = F.dropout(nodes, p=self.dropout, training=self.training)
            nodes = F.normalize(nodes, p=2, dim=1)
            layers.append(nodes)
        return self._split(torch.cat(layers, dim=1))


class GraphSAGE(GraphRecommender):
    """Hamilton et al. (2017), mean aggregator, trained full-graph.

    Each layer concatenates a node with the mean of its neighbours and passes the
    result through a linear layer. Uses the row-normalised adjacency, not the symmetric
    one, so every neighbour contributes equally regardless of its own degree.

    Trained over the whole graph rather than with neighbour sampling. Sampling in the
    original paper is an efficiency device for graphs too large to propagate over; at
    363,895 nodes that is unnecessary, and full-graph training removes a source of
    variance from the comparison. The inductive property comes from the aggregator
    consuming node features, not from the sampling.

    ``normalise_output`` follows the paper. It bounds scores to [-1, 1], which caps the
    margin BPR can achieve, so it is exposed as a switch rather than hard-coded.

    ``final_activation`` controls whether ReLU is applied to the last layer. With it on,
    every embedding lies in the non-negative orthant, so every score is forced positive
    and the model cannot express a strong negative preference. Turning it off is common
    for GNNs used as embedding models and is tested rather than assumed.
    """

    def __init__(self, *args, adjacency: torch.Tensor, n_layers: int = 2,
                 normalise_output: bool = True, final_activation: bool = True,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.adjacency = adjacency          # row-normalised, from graph.mean_adjacency
        self.n_layers = n_layers
        self.normalise_output = normalise_output
        self.final_activation = final_activation

        self.transforms = nn.ModuleList(
            [nn.Linear(2 * self.dim, self.dim) for _ in range(n_layers)])

    def propagate(self, nodes: torch.Tensor):
        for layer in range(self.n_layers):
            neighbours = torch.sparse.mm(self.adjacency, nodes)
            nodes = self.transforms[layer](torch.cat([nodes, neighbours], dim=1))
            last = layer == self.n_layers - 1
            if not last or self.final_activation:
                nodes = F.relu(nodes)
            if not last or self.normalise_output:
                nodes = F.normalize(nodes, p=2, dim=1)
        return self._split(nodes)