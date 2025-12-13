"""Data preprocessing utilities."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def train_val_test_split(
    interactions: pd.DataFrame,
    test_size: float = 0.1,
    val_size: float = 0.1,
    random_state: int | None = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split user interactions into train/validation/test ensuring each user appears in all sets.
    """
    train_rows = []
    val_rows = []
    test_rows = []

    for _, user_df in interactions.groupby("user_id"):
        if len(user_df) < 3:
            train_rows.append(user_df)
            continue
        train_temp, test = train_test_split(
            user_df, test_size=test_size, random_state=random_state
        )
        train, val = train_test_split(
            train_temp, test_size=val_size / (1 - test_size), random_state=random_state
        )
        train_rows.append(train)
        val_rows.append(val)
        test_rows.append(test)

    return (
        pd.concat(train_rows).reset_index(drop=True),
        pd.concat(val_rows).reset_index(drop=True),
        pd.concat(test_rows).reset_index(drop=True),
    )


def negative_sampling(
    interactions: pd.DataFrame,
    num_items: int,
    num_negatives: int = 1,
    random_state: int | None = 42,
) -> pd.DataFrame:
    """Generate negative samples for ranking models."""
    rng = np.random.default_rng(random_state)
    negatives = []
    grouped = interactions.groupby("user_id")
    for user_id, user_df in grouped:
        positive_items = set(user_df["item_id"].tolist())
        candidates = np.setdiff1d(np.arange(num_items), list(positive_items))
        if len(candidates) == 0:
            continue
        chosen_negatives = rng.choice(
            candidates,
            size=min(num_negatives * len(user_df), len(candidates)),
            replace=False,
        )
        for item_id in chosen_negatives:
            negatives.append((user_id, int(item_id), 0))

    return pd.concat(
        [interactions, pd.DataFrame(negatives, columns=["user_id", "item_id", "relevance"])]
    ).reset_index(drop=True)
