"""FAISS retrieval utilities and sweeps."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable

import faiss
import numpy as np


@dataclass
class SweepResult:
    nprobe: int
    latency_ms: float
    recall: float


def build_ivf_index(item_embeddings: np.ndarray, nlist: int = 50, random_state: int | None = 42) -> faiss.IndexIVFFlat:
    dim = item_embeddings.shape[1]
    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
    np.random.seed(random_state)
    faiss.normalize_L2(item_embeddings)
    index.train(item_embeddings)
    index.add(item_embeddings)
    return index


def search(index: faiss.IndexIVFFlat, queries: np.ndarray, k: int = 50, nprobe: int = 1) -> np.ndarray:
    faiss.normalize_L2(queries)
    index.nprobe = nprobe
    _, indices = index.search(queries, k)
    return indices


def brute_force_search(item_embeddings: np.ndarray, queries: np.ndarray, k: int = 50) -> np.ndarray:
    faiss.normalize_L2(item_embeddings)
    faiss.normalize_L2(queries)
    scores = queries @ item_embeddings.T
    top_k = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    row_indices = np.arange(scores.shape[0])[:, None]
    top_sorted = np.take_along_axis(top_k, np.argsort(-scores[row_indices, top_k]), axis=1)
    return top_sorted


def sweep_nprobe(
    index: faiss.IndexIVFFlat,
    item_embeddings: np.ndarray,
    queries: np.ndarray,
    k: int,
    nprobe_values: Iterable[int],
) -> list[SweepResult]:
    ground_truth = brute_force_search(item_embeddings.copy(), queries.copy(), k)
    results: list[SweepResult] = []

    for nprobe in nprobe_values:
        start = time.time()
        approx = search(index, queries.copy(), k=k, nprobe=nprobe)
        latency_ms = (time.time() - start) * 1000
        recall = (approx == ground_truth).sum() / ground_truth.size
        results.append(SweepResult(nprobe=nprobe, latency_ms=latency_ms, recall=recall))

    return results
