"""Synthetic interaction and metadata generator for the demo."""
from __future__ import annotations

import numpy as np
import pandas as pd


def generate_synthetic_data(
    num_users: int = 200,
    num_items: int = 500,
    interactions_per_user: int = 30,
    num_categories: int = 8,
    random_state: int | None = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create a small implicit-feedback dataset.

    Returns a tuple of (interactions, item_metadata).
    - interactions: columns [user_id, item_id, relevance]
    - item_metadata: columns [item_id, category, popularity]
    """
    rng = np.random.default_rng(random_state)

    # Item metadata
    categories = rng.integers(0, num_categories, size=num_items)
    popularity = rng.random(num_items)
    item_metadata = pd.DataFrame(
        {
            "item_id": np.arange(num_items),
            "category": categories,
            "popularity": popularity,
        }
    )

    # Generate user-item interactions biased by popularity and category match
    user_pref = rng.normal(0, 1, size=(num_users, num_categories))
    interactions = []
    for user_id in range(num_users):
        preferred_cat = rng.integers(0, num_categories)
        cat_scores = user_pref[user_id] + (np.arange(num_categories) == preferred_cat) * 0.8
        item_scores = cat_scores[categories] + popularity * 0.5
        probs = np.exp(item_scores)
        probs /= probs.sum()
        chosen_items = rng.choice(
            num_items, size=interactions_per_user, replace=False, p=probs
        )
        for item_id in chosen_items:
            interactions.append((user_id, int(item_id), 1))

    interactions_df = pd.DataFrame(interactions, columns=["user_id", "item_id", "relevance"])
    return interactions_df, item_metadata
