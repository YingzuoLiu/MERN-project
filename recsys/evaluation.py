"""Ranking evaluation metrics."""
from __future__ import annotations

import numpy as np


def recall_at_k(recommended: list[int], ground_truth: set[int], k: int) -> float:
    if k == 0:
        return 0.0
    hits = sum(1 for item in recommended[:k] if item in ground_truth)
    return hits / min(k, len(ground_truth) or 1)


def ndcg_at_k(recommended: list[int], ground_truth: set[int], k: int) -> float:
    dcg = 0.0
    for idx, item in enumerate(recommended[:k]):
        if item in ground_truth:
            dcg += 1 / np.log2(idx + 2)
    idcg = sum(1 / np.log2(i + 2) for i in range(min(len(ground_truth), k)))
    return dcg / idcg if idcg > 0 else 0.0
