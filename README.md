# Multi-stage Recommendation Demo

This repository contains a minimal, runnable Python demo of a three-stage recommendation stack:

1. **Recall:** a LightGCN-like placeholder that learns user/item embeddings with a BPR objective.
2. **Rank:** a small DeepFM/MLP-style classifier using dense embeddings + categorical metadata.
3. **Rerank:** a diversity-aware heuristic that discourages repeated categories.

Features include synthetic data generation, preprocessing utilities, evaluation metrics (Recall@K, NDCG@K), and a FAISS IVF latency/recall sweep over `nprobe` values.

## Quickstart

Run the end-to-end demo (data generation → training → evaluation → FAISS sweep) with one command:

```bash
pip install -r requirements.txt
python scripts/run_demo.py
```

Outputs include printed metric summaries and a FAISS sweep plot saved to `outputs/faiss_sweep.png`.

## Project Structure

```
recsys/
  data/
    synthetic.py      # synthetic interaction + metadata generator
    preprocessing.py  # train/val/test split and negative sampling
  models/
    recall.py         # LightGCN-style placeholder
    rank.py           # DeepFM/MLP-style ranker
    rerank.py         # diversity heuristic
  evaluation.py       # Recall@K and NDCG@K metrics
  faiss_search.py    # FAISS IVF index builder and nprobe sweep
scripts/
  run_demo.py         # orchestrates the full pipeline
requirements.txt       # lightweight dependencies
outputs/               # generated plots and artifacts
```

The code is extensively commented to serve as a teaching aid for how multi-stage recommendation stacks fit together without heavy dependencies.
