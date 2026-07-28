"""Utilities for constructing and validating bipartite user-item graphs.

Users and items share one node index space. User ``u`` is node ``u``, while
item ``i`` is node ``n_users + i``.

Graphs are constructed only from training interactions. Validation and test
partitions are used only for leakage checks.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
import torch

from pathlib import Path
import pyarrow.parquet as pq

__all__ = [
    "build_edge_index",
    "node_degrees",
    "normalised_adjacency",
    "spectral_radius",
    "verify_symmetry",
    "count_held_out_edges",
    "graph_statistics",
]


def build_edge_index(
    train: pd.DataFrame,
    n_users: int,
    n_items: int,
) -> torch.Tensor:
    """Edge index for the undirected bipartite graph, shape (2, 2E).

    Each interaction becomes two entries, user->item and item->user, so the
    adjacency comes out symmetric. Item nodes are offset by n_users.

    No self-loops. GCN adds them, LightGCN deliberately does not.
    """
    users = train["user_idx"].to_numpy().astype(np.int64)
    items = train["item_idx"].to_numpy().astype(np.int64) + n_users

    if users.max(initial=-1) >= n_users:
        raise ValueError("user index out of range")
    if items.max(initial=-1) >= n_users + n_items:
        raise ValueError("item index out of range")

    src = np.concatenate([users, items])
    dst = np.concatenate([items, users])
    return torch.from_numpy(np.stack([src, dst]))


def node_degrees(edge_index: torch.Tensor, n_nodes: int) -> torch.Tensor:
    """Degree of every node, counted from the edge index."""
    return torch.bincount(edge_index[0], minlength=n_nodes).to(torch.float32)


def normalised_adjacency(
    edge_index: torch.Tensor,
    n_nodes: int,
    device: str = "cpu",
) -> torch.Tensor:
    """Construct the sparse adjacency ``D^-1/2 A D^-1/2``.

    Node degrees are clamped to avoid division by zero for isolated nodes.
    """
    deg = node_degrees(edge_index, n_nodes).clamp(min=1.0)
    inv_sqrt = deg.pow(-0.5)

    src, dst = edge_index[0], edge_index[1]
    values = inv_sqrt[src] * inv_sqrt[dst]

    adj = torch.sparse_coo_tensor(
        edge_index, values, size=(n_nodes, n_nodes), device=device
    )
    return adj.coalesce()


def spectral_radius(
    adj: torch.Tensor,
    max_iterations: int = 500,
    tol: float = 1e-7,
    seed: int = 0,
) -> tuple[float, int]:
    """Estimate the spectral radius using power iteration.

    Iteration stops when the norm estimate changes by less than ``tol``.

    Returns:
        A tuple containing the estimate and number of iterations used.
    """
    
    generator = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(adj.shape[0], 1, generator=generator).to(adj.device)
    x = x / x.norm()

    previous = 0.0
    for step in range(1, max_iterations + 1):
        y = torch.sparse.mm(adj, x)
        norm = float(y.norm())
        if norm == 0:
            return 0.0, step
        x = y / norm
        if abs(norm - previous) < tol:
            return norm, step
        previous = norm

    return previous, max_iterations


def verify_symmetry(edge_index: torch.Tensor) -> bool:
    """Return whether every edge has a reverse edge."""
    forward = (edge_index[0].to(torch.int64) << 32) + edge_index[1].to(torch.int64)
    backward = (edge_index[1].to(torch.int64) << 32) + edge_index[0].to(torch.int64)
    return bool(torch.equal(torch.sort(forward).values, torch.sort(backward).values))


def count_held_out_edges(
    edge_index: torch.Tensor,
    held_out: pd.DataFrame,
    n_users: int,
) -> int:
    """Count held-out interactions that also appear in the graph."""
    # Only user-to-item edges are needed because the graph is bidirectional.
    mask = edge_index[0] < n_users          
    graph_pairs = (edge_index[0][mask].to(torch.int64) << 32) \
        + (edge_index[1][mask].to(torch.int64) - n_users)

    held = (torch.from_numpy(held_out["user_idx"].to_numpy().astype(np.int64)) << 32) \
        + torch.from_numpy(held_out["item_idx"].to_numpy().astype(np.int64))

    graph_sorted = torch.sort(graph_pairs).values
    positions = torch.searchsorted(graph_sorted, torch.sort(held).values)
    positions = positions.clamp(max=graph_sorted.numel() - 1)
    return int((graph_sorted[positions] == torch.sort(held).values).sum())


def graph_statistics(
    edge_index: torch.Tensor,
    n_users: int,
    n_items: int,
) -> Dict[str, float]:
    """Node/edge counts and degree summaries."""
    n_nodes = n_users + n_items
    deg = node_degrees(edge_index, n_nodes)
    user_deg, item_deg = deg[:n_users], deg[n_users:]

    return {
        "nodes": int(n_nodes),
        "users": int(n_users),
        "items": int(n_items),
        "interactions": int(edge_index.shape[1] // 2),
        "directed_edges": int(edge_index.shape[1]),
        "isolated_nodes": int((deg == 0).sum()),
        "mean_user_degree": float(user_deg.mean()),
        "mean_item_degree": float(item_deg.mean()),
        "min_user_degree": float(user_deg.min()),
        "min_item_degree": float(item_deg.min()),
        "max_user_degree": float(user_deg.max()),
        "max_item_degree": float(item_deg.max()),
    }

def connected_components(
    edge_index: torch.Tensor,
    n_nodes: int,
    max_iterations: int = 200,
) -> torch.Tensor:
    """Assign each node the smallest node ID in its component.

    Component labels are found through iterative minimum-label propagation.
    """
    labels = torch.arange(n_nodes, device=edge_index.device, dtype=torch.int64)
    src, dst = edge_index[0], edge_index[1]

    for _ in range(max_iterations):
        updated = labels.clone()
        updated.scatter_reduce_(0, src, labels[dst], reduce="amin", include_self=True)
        if torch.equal(updated, labels):
            return labels
        labels = updated

    raise RuntimeError(f"components did not converge in {max_iterations} passes")


def load_graph(subsets_dir, subset: str, protocol: str) -> Dict[str, object]:
    """Rebuild a graph from a saved training split."""

    base = Path(subsets_dir) / subset
    train = pq.read_table(
        base / "splits" / protocol / "train.parquet",
        columns=["user_idx", "item_idx"],
    ).to_pandas()

    n_users = len(pq.read_table(base / "user_mapping.parquet"))
    n_items = len(pq.read_table(base / "item_mapping.parquet"))

    return {
        "edge_index": build_edge_index(train, n_users, n_items),
        "n_users": n_users,
        "n_items": n_items,
        "n_nodes": n_users + n_items,
    }