"""LightGCN-like placeholder using simple matrix factorization with BPR loss."""
from __future__ import annotations

import numpy as np
import pandas as pd


class LightGCNPlaceholder:
    """
    Minimal implicit-feedback model that learns user/item embeddings via a simplified
    Bayesian Personalized Ranking (BPR) objective.
    """

    def __init__(self, num_users: int, num_items: int, embedding_dim: int = 32, lr: float = 0.05, reg: float = 1e-4, epochs: int = 10, negatives_per_pos: int = 3, random_state: int | None = 42):
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.lr = lr
        self.reg = reg
        self.epochs = epochs
        self.negatives_per_pos = negatives_per_pos
        self.random_state = random_state
        self.user_embeddings: np.ndarray | None = None
        self.item_embeddings: np.ndarray | None = None

    def fit(self, interactions: pd.DataFrame) -> None:
        rng = np.random.default_rng(self.random_state)
        self.user_embeddings = 0.01 * rng.standard_normal((self.num_users, self.embedding_dim))
        self.item_embeddings = 0.01 * rng.standard_normal((self.num_items, self.embedding_dim))

        interactions_array = interactions[["user_id", "item_id"]].to_numpy(dtype=int)
        for _ in range(self.epochs):
            rng.shuffle(interactions_array)
            for user_id, pos_item in interactions_array:
                # sample negative items
                neg_items = rng.integers(0, self.num_items, size=self.negatives_per_pos)
                for neg_item in neg_items:
                    self._update(user_id, pos_item, int(neg_item))

    def _update(self, user_id: int, pos_item: int, neg_item: int) -> None:
        assert self.user_embeddings is not None and self.item_embeddings is not None
        u = self.user_embeddings[user_id]
        pos = self.item_embeddings[pos_item]
        neg = self.item_embeddings[neg_item]

        # score difference for BPR
        x_uij = np.dot(u, pos - neg)
        sigmoid = 1 / (1 + np.exp(-x_uij))

        grad = (1 - sigmoid)
        u_grad = grad * (pos - neg) - self.reg * u
        pos_grad = grad * u - self.reg * pos
        neg_grad = -grad * u - self.reg * neg

        self.user_embeddings[user_id] += self.lr * u_grad
        self.item_embeddings[pos_item] += self.lr * pos_grad
        self.item_embeddings[neg_item] += self.lr * neg_grad

    def recommend(self, user_id: int, k: int = 50, exclude_seen: set[int] | None = None) -> np.ndarray:
        assert self.user_embeddings is not None and self.item_embeddings is not None
        scores = self.item_embeddings @ self.user_embeddings[user_id]
        if exclude_seen:
            scores[list(exclude_seen)] = -np.inf
        top_indices = np.argpartition(-scores, kth=min(k, len(scores) - 1))[:k]
        return top_indices[np.argsort(-scores[top_indices])]

    def get_item_embeddings(self) -> np.ndarray:
        assert self.item_embeddings is not None
        return self.item_embeddings

    def get_user_embeddings(self) -> np.ndarray:
        assert self.user_embeddings is not None
        return self.user_embeddings
