"""Temporal train/validation/test splitting for the Amazon Reviews 2023 dataset.

This module implements the two data-splitting protocols published by the dataset
authors alongside the Amazon Reviews 2023 release, together with the diagnostic
and verification routines required to demonstrate that the resulting splits are
correct and free of structural leakage.

Two protocols are provided:

``leave_last_out_split``
    For each user, the chronologically final interaction is held out for testing,
    the penultimate interaction for validation, and all earlier interactions are
    used for training. This is the protocol used by the large majority of the
    recommendation literature and by the official ``last_out`` benchmark files.
    It respects each user's *local* timeline but not the *global* timeline, and
    therefore admits the form of data leakage analysed by Ji et al. (2023).

``global_timestamp_split``
    All interactions are partitioned by two absolute cut-off timestamps that are
    fixed across every product category in the released benchmark. This protocol
    respects the global timeline and is leakage-free, at the cost of leaving some
    users with no training history at all.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "OFFICIAL_T1",
    "OFFICIAL_T2",
    "SORT_KEYS",
    "SplitResult",
    "canonical_sort_order",
    "sort_canonically",
    "count_timestamp_ties",
    "leave_last_out_split",
    "global_timestamp_split",
    "verify_partition",
    "verify_no_pair_overlap",
    "verify_local_timeline",
    "verify_global_boundaries",
    "quantify_lookahead",
    "identify_cold_items",
    "users_without_training_history",
    "sample_evaluation_users",
]



# Constants


# Official global-timestamp cut-offs (milliseconds since the Unix epoch).
# These values are fixed by the dataset authors and are identical across all
# product categories, which is what makes results directly comparable between studies. 

OFFICIAL_T1: int = 1_628_643_414_042  # 2021-08-11 00:56:54.042 UTC
OFFICIAL_T2: int = 1_658_002_729_837  # 2022-07-16 20:18:49.837 UTC

# Canonical ordering used everywhere in the project.
#
# ``timestamp`` alone is not a total order: a user may review several products
# within the same millisecond, in which case "the final interaction" would be
# decided by whatever order the rows happen to occupy in memory, and the split
# would not be reproducible across runs. Appending ``item_idx`` as a final key
# makes the ordering deterministic. The choice of ``item_idx`` is arbitrary.

SORT_KEYS: Sequence[str] = ("user_idx", "timestamp", "item_idx")



# Container


@dataclass(frozen=True)
class SplitResult:
    """A train/validation/test partition of an interaction table."""

    name: str
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame

    @property
    def counts(self) -> Dict[str, int]:
        return {
            "train": len(self.train),
            "validation": len(self.validation),
            "test": len(self.test),
            "total": len(self.train) + len(self.validation) + len(self.test),
        }

    def summary(self) -> pd.DataFrame:
        """Row counts plus user/item coverage for each partition."""
        rows = []
        for part_name in ("train", "validation", "test"):
            part = getattr(self, part_name)
            rows.append(
                {
                    "split": self.name,
                    "partition": part_name,
                    "interactions": len(part),
                    "users": part["user_idx"].nunique(),
                    "items": part["item_idx"].nunique(),
                }
            )
        return pd.DataFrame(rows)



# Ordering


def canonical_sort_order(interactions: pd.DataFrame) -> np.ndarray:
    """Return the indices that sort the interactions consistently.
    The data is sorted by user_idx, timestamp, and item_idx.
    Since np.lexsort uses the last key as the main sorting key,
    the keys are passed in reverse order.
    """
    return np.lexsort(
        (
            interactions["item_idx"].to_numpy(),
            interactions["timestamp"].to_numpy(),
            interactions["user_idx"].to_numpy(),
        )
    )


def sort_canonically(interactions: pd.DataFrame) -> pd.DataFrame:
    """Return ``interactions`` in canonical order with a reset index."""
    order = canonical_sort_order(interactions)
    return interactions.iloc[order].reset_index(drop=True)


def _block_end_positions(user_idx: np.ndarray) -> np.ndarray:
    """Positions of the final row of each user's block in a sorted array."""
    if user_idx.size == 0:
        return np.empty(0, dtype=np.int64)
    is_last = np.empty(user_idx.size, dtype=bool)
    is_last[-1] = True
    is_last[:-1] = user_idx[:-1] != user_idx[1:]
    return np.flatnonzero(is_last)



# Diagnostics


def count_timestamp_ties(interactions: pd.DataFrame) -> Dict[str, int]:
    """Count users affected by identical timestamps.

    For leave-last-out splitting, ties at the final or penultimate
    interaction are resolved using item_idx.

    Returns counts for test ties, validation ties, and affected users.
    """

    ordered = sort_canonically(interactions)
    users = ordered["user_idx"].to_numpy()
    stamps = ordered["timestamp"].to_numpy()

    last_pos = _block_end_positions(users)
    if last_pos.size == 0:
        return {
            "n_users": 0,
            "test_position_ties": 0,
            "validation_position_ties": 0,
            "users_affected": 0,
        }

    counts = np.diff(np.concatenate(([-1], last_pos)))
    if counts.min() < 3:
        raise ValueError(
            "count_timestamp_ties requires at least 3 interactions per user; "
            f"minimum observed was {counts.min()}."
        )

    test_pos = last_pos
    valid_pos = last_pos - 1
    third_pos = last_pos - 2

    test_tie = stamps[test_pos] == stamps[valid_pos]
    valid_tie = stamps[valid_pos] == stamps[third_pos]

    return {
        "n_users": int(last_pos.size),
        "test_position_ties": int(test_tie.sum()),
        "validation_position_ties": int(valid_tie.sum()),
        "users_affected": int((test_tie | valid_tie).sum()),
    }



# Splitting protocols



def leave_last_out_split(interactions: pd.DataFrame) -> SplitResult:
    """Partition by the official ``last_out`` protocol.

    For a user with ``N`` chronologically ordered interactions: the first
    ``N - 2`` form training data, the ``N-1``-th is validation and the ``N``-th
    is test. Every user therefore contributes exactly one validation and one
    test interaction.

    Requires at least three interactions per user. The 5-core property of the
    processed dataset guarantees at least five.
    """
    ordered = sort_canonically(interactions)
    users = ordered["user_idx"].to_numpy()

    last_pos = _block_end_positions(users)
    counts = np.diff(np.concatenate(([-1], last_pos)))
    if counts.size and counts.min() < 3:
        raise ValueError(
            "leave_last_out_split requires at least 3 interactions per user; "
            f"minimum observed was {counts.min()}."
        )

    test_pos = last_pos
    valid_pos = last_pos - 1

    held_out = np.zeros(len(ordered), dtype=bool)
    held_out[test_pos] = True
    held_out[valid_pos] = True

    return SplitResult(
        name="last_out",
        train=ordered.loc[~held_out].reset_index(drop=True),
        validation=ordered.iloc[valid_pos].reset_index(drop=True),
        test=ordered.iloc[test_pos].reset_index(drop=True),
    )


def global_timestamp_split(
    interactions: pd.DataFrame,
    t1: int = OFFICIAL_T1,
    t2: int = OFFICIAL_T2,
) -> SplitResult:
    """Split interactions using absolute timestamp boundaries.

    Training contains interactions before t1, validation contains interactions
    from t1 up to t2, and test contains interactions from t2 onward.

    Unlike leave-last-out, this split follows the global timeline, so training
    interactions do not occur after test interactions.

    A limitation is that users with no interactions before t1 will have no
    training history and may not be usable by the model.
    """

    if not t1 < t2:
        raise ValueError(f"t1 must precede t2 (got t1={t1}, t2={t2}).")

    ordered = sort_canonically(interactions)
    stamps = ordered["timestamp"].to_numpy()

    train_mask = stamps < t1
    valid_mask = (stamps >= t1) & (stamps < t2)
    test_mask = stamps >= t2

    return SplitResult(
        name="timestamp",
        train=ordered.loc[train_mask].reset_index(drop=True),
        validation=ordered.loc[valid_mask].reset_index(drop=True),
        test=ordered.loc[test_mask].reset_index(drop=True),
    )



# Verification



def verify_partition(result: SplitResult, expected_total: int) -> Dict[str, object]:
    """Check that the three partitions tile the input exactly once."""
    counts = result.counts
    return {
        "expected_total": int(expected_total),
        "observed_total": int(counts["total"]),
        "is_exact_partition": bool(counts["total"] == expected_total),
        **{k: int(v) for k, v in counts.items()},
    }


def _pair_keys(frame: pd.DataFrame) -> np.ndarray:
    """Encode (user_idx, item_idx) pairs as single integers for set operations."""
    users = frame["user_idx"].to_numpy(dtype=np.int64)
    items = frame["item_idx"].to_numpy(dtype=np.int64)
    return users * np.int64(1 << 32) + items


def verify_no_pair_overlap(result: SplitResult) -> Dict[str, int]:
    """Confirm that no (user, item) pair appears in more than one partition."""

    
    
    train = _pair_keys(result.train)
    valid = _pair_keys(result.validation)
    test = _pair_keys(result.test)
    return {
        "train_validation_overlap": int(np.intersect1d(train, valid).size),
        "train_test_overlap": int(np.intersect1d(train, test).size),
        "validation_test_overlap": int(np.intersect1d(valid, test).size),
    }


def verify_local_timeline(result: SplitResult) -> Dict[str, int]:
    """Check that each user's interactions remain in chronological order.

    For users appearing in multiple splits, the latest training timestamp
    should be before or equal to the validation timestamp, and the validation
    timestamp should be before or equal to the test timestamp.

    Equal timestamps are allowed because item_idx is used to break ties.
    """
    train_max = result.train.groupby("user_idx")["timestamp"].max()
    valid_max = result.validation.groupby("user_idx")["timestamp"].max()
    test_min = result.test.groupby("user_idx")["timestamp"].min()

    tv = pd.concat([train_max, valid_max], axis=1, join="inner", keys=["train", "valid"])
    vt = pd.concat([valid_max, test_min], axis=1, join="inner", keys=["valid", "test"])
    tt = pd.concat([train_max, test_min], axis=1, join="inner", keys=["train", "test"])

    return {
        "users_compared_train_validation": int(len(tv)),
        "violations_train_after_validation": int((tv["train"] > tv["valid"]).sum()),
        "users_compared_validation_test": int(len(vt)),
        "violations_validation_after_test": int((vt["valid"] > vt["test"]).sum()),
        "users_compared_train_test": int(len(tt)),
        "violations_train_after_test": int((tt["train"] > tt["test"]).sum()),
    }


def verify_global_boundaries(
    result: SplitResult,
    t1: int = OFFICIAL_T1,
    t2: int = OFFICIAL_T2,
) -> Dict[str, object]:
    """Confirm the absolute cut-offs are respected by every partition."""
    train_ts = result.train["timestamp"].to_numpy()
    valid_ts = result.validation["timestamp"].to_numpy()
    test_ts = result.test["timestamp"].to_numpy()

    return {
        "train_max": int(train_ts.max()) if train_ts.size else None,
        "validation_min": int(valid_ts.min()) if valid_ts.size else None,
        "validation_max": int(valid_ts.max()) if valid_ts.size else None,
        "test_min": int(test_ts.min()) if test_ts.size else None,
        "train_before_t1": bool(train_ts.size == 0 or train_ts.max() < t1),
        "validation_within_bounds": bool(
            valid_ts.size == 0 or (valid_ts.min() >= t1 and valid_ts.max() < t2)
        ),
        "test_after_t2": bool(test_ts.size == 0 or test_ts.min() >= t2),
    }



# Leakage quantification



def quantify_lookahead(
    train: pd.DataFrame,
    test: pd.DataFrame,
    percentiles: Sequence[float] = (25, 50, 75, 90, 99),
) -> Dict[str, object]:
    """Measure how much future training data exists after each test interaction.

    For each test timestamp, count the training interactions that occur later
    in the global timeline. A global timestamp split should produce zero
    lookahead, while leave-last-out may contain future interactions.

    Returns summary statistics, the counts for each test interaction,
    and the counts as fractions of the training set.
    """
    train_ts = np.sort(train["timestamp"].to_numpy())
    test_ts = test["timestamp"].to_numpy()
    n_train = train_ts.size

    # Number of training rows strictly after each test timestamp.
    n_future = n_train - np.searchsorted(train_ts, test_ts, side="right")
    fraction = n_future / n_train if n_train else np.zeros_like(n_future, dtype=float)

    pct = {
        f"p{int(p)}": float(np.percentile(n_future, p)) for p in percentiles
    }

    return {
        "n_test_instances": int(test_ts.size),
        "n_train_interactions": int(n_train),
        "test_instances_with_future_training_data": int((n_future > 0).sum()),
        "share_of_test_instances_affected": float((n_future > 0).mean()) if n_future.size else 0.0,
        "mean_future_interactions": float(n_future.mean()) if n_future.size else 0.0,
        "median_future_interactions": float(np.median(n_future)) if n_future.size else 0.0,
        "max_future_interactions": int(n_future.max()) if n_future.size else 0,
        "mean_share_of_training_set": float(fraction.mean()) if fraction.size else 0.0,
        "percentiles": pct,
        "per_test_counts": n_future,
    }


# Coverage


def identify_cold_items(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Items appearing in ``test`` with no training interactions whatsoever.

    These are strictly cold items: a transductive model has no learned embedding
    for them, so they can only be scored from side information. They form the
    evaluation population for the item cold-start experiment.
    """
    train_items = np.unique(train["item_idx"].to_numpy())
    test_items = np.unique(test["item_idx"].to_numpy())
    return np.setdiff1d(test_items, train_items, assume_unique=True)


def users_without_training_history(
    all_users: np.ndarray, train: pd.DataFrame
) -> np.ndarray:
    """Users with no training interactions at all.

    Under the global timestamp split these are users whose entire history falls
    after ``t1``. No model in this project can represent them: there are no
    interactions from which to learn an embedding, and no user-side features to
    fall back on. They are therefore excluded from evaluation and reported.
    """
    trained = np.unique(train["user_idx"].to_numpy())
    return np.setdiff1d(np.unique(all_users), trained, assume_unique=True)



# Evaluation sampling



def sample_evaluation_users(
    test: pd.DataFrame,
    train: pd.DataFrame,
    n_users: int = 50_000,
    seed: int = 42,
) -> np.ndarray:
    """Select a reproducible sample of users for evaluation.

    Only users that appear in the test set and have at least one training
    interaction are eligible. Users are sampled instead of items to avoid
    biasing the ranking metrics.

    The returned user IDs are sorted so all models are evaluated using
    the same user sample.
    """
    if n_users <= 0:
        raise ValueError(f"n_users must be positive (got {n_users}).")

    test_users = np.unique(test["user_idx"].to_numpy())
    train_users = np.unique(train["user_idx"].to_numpy())
    eligible = np.intersect1d(test_users, train_users, assume_unique=True)

    if eligible.size <= n_users:
        return eligible

    rng = np.random.default_rng(seed)
    chosen = rng.choice(eligible, size=n_users, replace=False)
    chosen.sort()
    return chosen
