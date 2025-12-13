"""End-to-end multi-stage recommendation demo with FAISS sweep."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from recsys.data.preprocessing import negative_sampling, train_val_test_split
from recsys.data.synthetic import generate_synthetic_data
from recsys.evaluation import ndcg_at_k, recall_at_k
from recsys.faiss_search import build_ivf_index, sweep_nprobe
from recsys.models.rank import DeepFMPlaceholder
from recsys.models.recall import LightGCNPlaceholder
from recsys.models.rerank import diversify_rerank


OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def run_pipeline(seed: int = 42) -> dict[str, float]:
    interactions, item_metadata = generate_synthetic_data(random_state=seed)
    num_users = interactions["user_id"].nunique()
    num_items = interactions["item_id"].nunique()

    # Train/validation/test split
    train, val, test = train_val_test_split(interactions, random_state=seed)

    # Train recall model
    recall_model = LightGCNPlaceholder(num_users, num_items, random_state=seed)
    recall_model.fit(train)

    # Prepare ranking data with negative samples
    ranking_data = negative_sampling(pd.concat([train, val]), num_items=num_items)
    ranker = DeepFMPlaceholder(random_state=seed)
    ranker.fit(
        ranking_data,
        recall_model.get_user_embeddings(),
        recall_model.get_item_embeddings(),
        item_metadata,
    )

    # Evaluation on test set
    metrics = []
    k = 10
    for user_id, user_df in test.groupby("user_id"):
        seen = set(pd.concat([train, val]).query("user_id == @user_id")["item_id"].tolist())
        recall_candidates = recall_model.recommend(user_id, k=5 * k, exclude_seen=seen)
        # Score candidates with ranker
        user_ids = np.full_like(recall_candidates, user_id)
        scores = ranker.predict_scores(
            user_ids,
            recall_candidates,
            recall_model.get_user_embeddings(),
            recall_model.get_item_embeddings(),
            item_metadata,
        )
        ranked = recall_candidates[np.argsort(-scores)]
        reranked = diversify_rerank(ranked.tolist(), item_metadata, top_k=k)
        ground_truth = set(user_df["item_id"].tolist())
        metrics.append(
            {
                "user_id": user_id,
                "recall": recall_at_k(reranked, ground_truth, k),
                "ndcg": ndcg_at_k(reranked, ground_truth, k),
            }
        )

    metrics_df = pd.DataFrame(metrics)
    summary = {"recall@10": metrics_df["recall"].mean(), "ndcg@10": metrics_df["ndcg"].mean()}

    # FAISS latency/recall sweep using item embeddings and user queries
    item_embeddings = recall_model.get_item_embeddings().astype(np.float32)
    user_embeddings = recall_model.get_user_embeddings().astype(np.float32)
    index = build_ivf_index(item_embeddings.copy(), nlist=64)
    queries = user_embeddings[np.random.choice(num_users, size=50, replace=False)]
    sweep = sweep_nprobe(index, item_embeddings.copy(), queries.copy(), k=20, nprobe_values=[1, 2, 4, 8, 16, 32])

    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot([r.nprobe for r in sweep], [r.recall for r in sweep], marker="o", label="Recall")
    ax1.set_xlabel("nprobe")
    ax1.set_ylabel("Recall", color="C0")
    ax2 = ax1.twinx()
    ax2.plot([r.nprobe for r in sweep], [r.latency_ms for r in sweep], marker="s", color="C1", label="Latency")
    ax2.set_ylabel("Latency (ms)", color="C1")
    fig.suptitle("FAISS IVF nprobe trade-off")
    fig.tight_layout()
    fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.85))
    plot_path = OUTPUT_DIR / "faiss_sweep.png"
    fig.savefig(plot_path, dpi=150)

    print("\n=== Evaluation ===")
    print(metrics_df.describe())
    print("\nAverages:", summary)
    print("\nFAISS sweep:")
    for r in sweep:
        print(f"nprobe={r.nprobe:>2} | recall={r.recall:.3f} | latency_ms={r.latency_ms:.2f}")
    print(f"Plot saved to {plot_path}")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multi-stage recommendation demo")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()
    run_pipeline(seed=args.seed)


if __name__ == "__main__":
    main()
