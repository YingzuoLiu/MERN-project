"""Simple DeepFM/MLP-style placeholder using scikit-learn."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder


class DeepFMPlaceholder:
    """
    Combines dense user/item embeddings with item categorical features and trains a
    small MLP classifier to predict click/relevance.
    """

    def __init__(self, hidden_units: tuple[int, ...] = (64, 32), random_state: int | None = 42):
        self.hidden_units = hidden_units
        self.random_state = random_state
        self.model: MLPClassifier | None = None
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    def _build_features(
        self,
        interactions: pd.DataFrame,
        user_embeddings: np.ndarray,
        item_embeddings: np.ndarray,
        item_metadata: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray]:
        meta = item_metadata.set_index("item_id")
        categories = self.encoder.fit_transform(meta[["category"]])
        item_meta_emb = dict(zip(meta.index, categories))

        features = []
        labels = interactions["relevance"].to_numpy()
        for _, row in interactions.iterrows():
            user_vec = user_embeddings[int(row.user_id)]
            item_vec = item_embeddings[int(row.item_id)]
            cat_vec = item_meta_emb[int(row.item_id)]
            # Concatenate dense and one-hot features to mimic DeepFM/MLP inputs
            features.append(np.concatenate([user_vec, item_vec, cat_vec]))

        return np.stack(features), labels

    def fit(
        self,
        interactions: pd.DataFrame,
        user_embeddings: np.ndarray,
        item_embeddings: np.ndarray,
        item_metadata: pd.DataFrame,
    ) -> None:
        X, y = self._build_features(interactions, user_embeddings, item_embeddings, item_metadata)
        self.model = MLPClassifier(
            hidden_layer_sizes=self.hidden_units,
            activation="relu",
            max_iter=30,
            random_state=self.random_state,
        )
        self.model.fit(X, y)

    def predict_scores(
        self,
        user_ids: np.ndarray,
        item_ids: np.ndarray,
        user_embeddings: np.ndarray,
        item_embeddings: np.ndarray,
        item_metadata: pd.DataFrame,
    ) -> np.ndarray:
        assert self.model is not None
        meta = item_metadata.set_index("item_id")
        cat_enc = self.encoder.transform(meta[["category"]])
        item_meta_emb = dict(zip(meta.index, cat_enc))

        features = []
        for user_id, item_id in zip(user_ids, item_ids):
            user_vec = user_embeddings[int(user_id)]
            item_vec = item_embeddings[int(item_id)]
            cat_vec = item_meta_emb[int(item_id)]
            features.append(np.concatenate([user_vec, item_vec, cat_vec]))

        return self.model.predict_proba(np.stack(features))[:, 1]
