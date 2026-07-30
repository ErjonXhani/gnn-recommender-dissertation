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
    patience: int = 5,
    eval_every: int = 10,
    selection_metric: str = "ndcg@10",
    device: str = "cuda",
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, object]:
    """Train a model with BPR and select the best epoch using validation performance.

    Negatives are sampled uniformly without rejecting known positives. Users have roughly
    7 training interactions from a catalogue of 105,298 items, so the expected collision
    rate is below 0.01% and rejection sampling would add unnecessary cost.

    L2 regularisation is applied only to the embeddings involved in each batch. Optimiser
    weight decay would also update embeddings absent from the batch, producing a different
    objective for sparse embedding tables.
    """
    model = model.to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    generator = torch.Generator(device=device).manual_seed(seed)

    users = torch.as_tensor(train["user_idx"].to_numpy(), dtype=torch.long, device=device)
    items = torch.as_tensor(train["item_idx"].to_numpy(), dtype=torch.long, device=device)
    n_train = len(users)

    history = []
    best_score, best_epoch, best_state = -np.inf, -1, None
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
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

            regularisation = l2_reg * (
                e_u.pow(2).sum() + e_pos.pow(2).sum() + e_neg.pow(2).sum()
            ) / len(batch)

            optimiser.zero_grad(set_to_none=True)
            (loss + regularisation).backward()
            optimiser.step()

            epoch_loss += float(loss) * len(batch)

        record = {"epoch": epoch, "loss": epoch_loss / n_train,
                  "seconds": time.time() - t0}

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

            if epochs_without_improvement >= patience:
                if verbose:
                    print(f"  stopping: no improvement for {patience} evaluations")
                break

        history.append(record)

    if best_state is not None:
        model.load_state_dict(best_state)

    return {"model": model, "history": pd.DataFrame(history),
            "best_epoch": best_epoch, "best_val_score": best_score}