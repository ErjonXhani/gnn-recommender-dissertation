"""Top-K evaluation over the full item catalogue.

Every reported number in the project comes through here. A mistake in this file does
not raise an error, but it returns a plausible but wrong metric. So, the design favours
being checkable over being clever, and Notebook 08 validates it against a popularity
baseline before any real model is run.

Users are sampled but items never are. Krichene and Rendle (KDD 2020) show that
sampling items produces metrics inconsistent with their exact versions, to the point of
reversing the relative order of models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Sequence

import numpy as np
import pandas as pd
import torch

__all__ = ["build_csr", "Evaluator", "metrics_to_frame"]


def build_csr(frame: pd.DataFrame, n_users: int) -> tuple[np.ndarray, np.ndarray]:
    """Compress a (user_idx, item_idx) table into CSR form.

    Returns ``(indptr, indices)`` where user u owns ``indices[indptr[u]:indptr[u+1]]``.
    Lets a user's items be looked up without scanning the frame.
    """
    users = frame["user_idx"].to_numpy()
    items = frame["item_idx"].to_numpy()

    order = np.argsort(users, kind="stable")
    counts = np.bincount(users, minlength=n_users)

    indptr = np.zeros(n_users + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr[1:])
    return indptr, items[order].astype(np.int64)


def _gather_rows(
    indptr: np.ndarray,
    indices: np.ndarray,
    users: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Flatten the CSR rows for ``users`` into (batch_row, item) pairs.

    Ragged gather without a Python loop: every output position is mapped back to its
    source position in ``indices`` by a per-row constant offset.
    """
    starts = indptr[users]
    counts = indptr[users + 1] - starts
    total = int(counts.sum())
    if total == 0:
        return np.empty(0, np.int64), np.empty(0, np.int64)

    block_starts = np.zeros(len(users), dtype=np.int64)
    np.cumsum(counts[:-1], out=block_starts[1:])

    rows = np.repeat(np.arange(len(users), dtype=np.int64), counts)
    offsets = np.repeat(starts - block_starts, counts)
    cols = indices[np.arange(total, dtype=np.int64) + offsets]
    return rows, cols


@dataclass
class Evaluator:
    """Full-catalogue top-K evaluation for a fixed set of users.

    ``ground_truth`` holds the items each user should be rewarded for retrieving.
    ``mask`` holds items removed from the candidate pool before ranking,  anything the
    user has already interacted with. Recommending something a user already has is not
    a hit, and leaving those in inflates every metric.
    """

    eval_users: np.ndarray
    truth_indptr: np.ndarray
    truth_indices: np.ndarray
    mask_indptr: np.ndarray
    mask_indices: np.ndarray
    n_items: int
    k_values: Sequence[int] = (10, 20)
    device: str = "cuda"
    batch_users: int = 1024

    @classmethod
    def from_frames(
        cls,
        ground_truth: pd.DataFrame,
        mask_frames: Sequence[pd.DataFrame],
        eval_users: np.ndarray,
        n_users: int,
        n_items: int,
        k_values: Sequence[int] = (10, 20),
        device: str = "cuda",
        batch_users: int = 1024,
    ) -> "Evaluator":
        truth_indptr, truth_indices = build_csr(ground_truth, n_users)
        mask_indptr, mask_indices = build_csr(pd.concat(mask_frames), n_users)
        return cls(
            eval_users=np.asarray(eval_users),
            truth_indptr=truth_indptr,
            truth_indices=truth_indices,
            mask_indptr=mask_indptr,
            mask_indices=mask_indices,
            n_items=n_items,
            k_values=tuple(k_values),
            device=device,
            batch_users=batch_users,
        )

    def __post_init__(self):
        max_k = max(self.k_values)
        positions = torch.arange(1, max_k + 1, dtype=torch.float32, device=self.device)
        # position i (1-based) contributes 1 / log2(i + 1)
        self._discounts = 1.0 / torch.log2(positions + 1.0)
        # ideal DCG for n relevant items is the sum of the first n discounts
        self._ideal = torch.cat([
            torch.zeros(1, device=self.device),
            torch.cumsum(self._discounts, dim=0),
        ])

    @torch.no_grad()
    def evaluate(
        self,
        score_fn: Callable[[torch.Tensor], torch.Tensor],
        return_per_user: bool = True,
    ) -> Dict[str, object]:
        """Score every evaluation user against all items and compute metrics.

        ``score_fn`` takes a LongTensor of user ids and returns a
        ``(len(users), n_items)`` float tensor of scores on the same device.

        Per-user values are kept because the significance tests later compare models
        user by user, which needs the individual scores rather than the means.
        """
        max_k = max(self.k_values)
        per_user = {f"{m}@{k}": [] for k in self.k_values
                    for m in ("hit_rate", "recall", "ndcg", "mrr")}

        for start in range(0, len(self.eval_users), self.batch_users):
            users = self.eval_users[start:start + self.batch_users]
            user_tensor = torch.as_tensor(users, dtype=torch.long, device=self.device)

            scores = score_fn(user_tensor)
            if scores.shape != (len(users), self.n_items):
                raise ValueError(
                    f"score_fn returned {tuple(scores.shape)}, "
                    f"expected {(len(users), self.n_items)}"
                )
            scores = scores.float()

            # remove already-seen items from the candidate pool
            mask_rows, mask_cols = _gather_rows(self.mask_indptr, self.mask_indices, users)
            if mask_rows.size:
                scores[
                    torch.as_tensor(mask_rows, device=self.device),
                    torch.as_tensor(mask_cols, device=self.device),
                ] = torch.finfo(scores.dtype).min

            top_items = torch.topk(scores, max_k, dim=1).indices

            # mark which of the ranked items are actually relevant
            truth_rows, truth_cols = _gather_rows(self.truth_indptr, self.truth_indices, users)
            relevant = torch.zeros_like(scores, dtype=torch.bool)
            relevant[
                torch.as_tensor(truth_rows, device=self.device),
                torch.as_tensor(truth_cols, device=self.device),
            ] = True

            hits = relevant.gather(1, top_items)
            n_relevant = relevant.sum(dim=1)

            for k in self.k_values:
                hits_k = hits[:, :k]
                n_hits = hits_k.sum(dim=1)

                dcg = (hits_k.float() * self._discounts[:k]).sum(dim=1)
                idcg = self._ideal[torch.clamp(n_relevant, max=k)]

                any_hit = hits_k.any(dim=1)
                first_hit = torch.argmax(hits_k.int(), dim=1)  # argmax returns 0 when no hit exists, so reciprocal rank is masked below
                reciprocal = torch.where(
                    any_hit, 1.0 / (first_hit + 1).float(), torch.zeros_like(dcg)
                )

                per_user[f"hit_rate@{k}"].append(any_hit.float().cpu())
                per_user[f"recall@{k}"].append(
                    (n_hits / torch.clamp(n_relevant, min=1)).cpu())
                per_user[f"ndcg@{k}"].append((dcg / torch.clamp(idcg, min=1e-12)).cpu())
                per_user[f"mrr@{k}"].append(reciprocal.cpu())

            del scores, relevant

        per_user = {k: torch.cat(v).numpy() for k, v in per_user.items()}
        result = {"means": {k: float(v.mean()) for k, v in per_user.items()},
                  "n_users": int(len(self.eval_users))}
        if return_per_user:
            result["per_user"] = per_user
        return result


def metrics_to_frame(results: Dict[str, Dict]) -> pd.DataFrame:
    """Turn {model_name: evaluate() output} into a comparison table."""
    return pd.DataFrame(
        {name: res["means"] for name, res in results.items()}
    ).T.sort_index(axis=1)