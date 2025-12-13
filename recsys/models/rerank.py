"""Diversity-aware reranker."""
from __future__ import annotations

import pandas as pd


def diversify_rerank(
    ranked_items: list[int],
    item_metadata: pd.DataFrame,
    lambda_diversity: float = 0.3,
    top_k: int = 20,
) -> list[int]:
    """
    Simple heuristic that penalizes repeated categories to encourage diversity.
    Iterates through the ranked list and attenuates scores when the category has
    already been selected.
    """
    meta = item_metadata.set_index("item_id")
    category_counts: dict[int, int] = {}
    reranked: list[tuple[int, float]] = []

    for item_id in ranked_items:
        category = int(meta.loc[item_id, "category"])
        penalty = lambda_diversity * category_counts.get(category, 0)
        score = 1 / (1 + penalty)
        reranked.append((item_id, score))
        category_counts[category] = category_counts.get(category, 0) + 1
        if len(reranked) >= top_k:
            break

    reranked.sort(key=lambda x: -x[1])
    return [item for item, _ in reranked]
