"""BPR training loop shared by every model in the project.

Bayesian Personalised Ranking (Rendle et al., 2009) optimises the pairwise ordering of
an observed item against an unobserved one, rather than trying to predict a rating.
That matches the top-K task, where only the ordering matters.
"""

from __future__ import annotations

import copy
import time
from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

__all__ = ["train_bpr"]


def train_bpr(
    model,
    train: pd.DataFrame,
    n_items: int,
    evaluator=None,
    epochs: int = 200,
    batch_size: int = 8192,
    learning_rate: float = 1e-3,
    l2_reg: float = 1e-4,
    sampling: str = "interaction",
    patience: int = 5,
    eval_every: int = 10,
    selection_metric: str = "ndcg@10",
    device: str = "cuda",
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, object]:
    """Train a model with BPR and select the best epoch using validation performance.

    ``sampling`` controls how training pairs are drawn. ``"interaction"`` iterates over
    the interaction table, so a user's number of updates is proportional to their degree.
    ``"user"`` draws a user uniformly and then one of their items, which is what the
    LightGCN reference implementation does and gives every user the same expected number
    of updates. The choice matters because the metrics are averaged per user.

    Negatives are sampled uniformly without rejecting known positives. On the Amazon
    data users hold roughly 7 items from a catalogue of 105,298, so the expected
    collision rate is under 0.01%; on Gowalla it is 0.07%. Rejection sampling would add
    cost for no measurable difference.

    L2 regularisation is applied only to the embeddings involved in each batch, and to
    the layer-0 embeddings rather than the propagated ones. Optimiser weight decay would
    also update embeddings absent from the batch, producing a different objective for
    sparse embedding tables.
    """

    if sampling not in ("interaction", "user"):
        raise ValueError(f"sampling must be 'interaction' or 'user', got {sampling!r}")
    
    model = model.to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    generator = torch.Generator(device=device).manual_seed(seed)

    users = torch.as_tensor(train["user_idx"].to_numpy(), dtype=torch.long, device=device)
    items = torch.as_tensor(train["item_idx"].to_numpy(), dtype=torch.long, device=device)
    n_train = len(users)

    if sampling == "user":
        # LightGCN samples a user uniformly, then one of that user's items, rather than
        # sampling interactions. Low-degree users get the same number of updates as
        # heavy ones, which matters because the metrics are averaged per user.
        order_by_user = np.argsort(train["user_idx"].to_numpy(), kind="stable")
        counts = np.bincount(train["user_idx"].to_numpy(), minlength=model.n_users)
        indptr = np.zeros(model.n_users + 1, dtype=np.int64)
        np.cumsum(counts, out=indptr[1:])

        indptr = torch.as_tensor(indptr, device=device)
        sorted_items = torch.as_tensor(
            train["item_idx"].to_numpy()[order_by_user], dtype=torch.long, device=device)
        active = torch.as_tensor(np.flatnonzero(counts > 0), dtype=torch.long, device=device)

    history = []
    best_score, best_epoch, best_state = -np.inf, -1, None
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()

        if sampling == "user":
            drawn = active[torch.randint(0, len(active), (n_train,),
                                         generator=generator, device=device)]
            starts, spans = indptr[drawn], indptr[drawn + 1] - indptr[drawn]
            picks = (torch.rand(n_train, generator=generator, device=device)
                     * spans).long().clamp_(max=spans - 1)
            users, items = drawn, sorted_items[starts + picks]

        order = torch.randperm(n_train, generator=generator, device=device)
        epoch_loss = 0.0

        for start in range(0, n_train, batch_size):
            batch = order[start:start + batch_size]
            batch_users, positives = users[batch], items[batch]
            negatives = torch.randint(
                0, n_items, (len(batch),), generator=generator, device=device)

            user_emb, item_emb = model()
            e_u = user_emb[batch_users]
            e_pos, e_neg = item_emb[positives], item_emb[negatives]

            difference = (e_u * e_pos).sum(1) - (e_u * e_neg).sum(1)
            loss = F.softplus(-difference).mean()

            # regularise the layer-0 embeddings rather than the propagated ones.
            # Those are the actual free parameters, and it is what LightGCN and NGCF do.
            base_users, base_items = model.base_embeddings()
            regularisation = l2_reg * (
                base_users[batch_users].pow(2).sum()
                + base_items[positives].pow(2).sum()
                + base_items[negatives].pow(2).sum()
            ) / len(batch)

            optimiser.zero_grad(set_to_none=True)
            (loss + regularisation).backward()
            optimiser.step()

            epoch_loss += float(loss) * len(batch)

        record = {"epoch": epoch, "loss": epoch_loss / n_train,
                  "seconds": time.time() - t0}

        should_stop = False
        if evaluator is not None and (epoch % eval_every == 0 or epoch == epochs):
            model.eval()
            means = evaluator.evaluate(model.score_fn(), return_per_user=False)["means"]
            record.update({f"val_{k}": v for k, v in means.items()})

            score = means[selection_metric]
            if score > best_score:
                best_score, best_epoch = score, epoch
                best_state = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if verbose:
                print(f"  epoch {epoch:>3}  loss {record['loss']:.4f}  "
                      f"val {selection_metric} {score:.5f}"
                      f"{'  *' if best_epoch == epoch else ''}", flush=True)

            should_stop = epochs_without_improvement >= patience

        history.append(record)

        if should_stop:
            if verbose:
                print(f"  stopping: no improvement for {patience} evaluations")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {"model": model, "history": pd.DataFrame(history),
            "best_epoch": best_epoch, "best_val_score": best_score}