#!/usr/bin/env python3

"""
BiEncoder Evaluator for comprehensive model comparison.

Computes 18 metrics:
  - Intrinsic: cosine similarity, mean/std/min/max
  - Ranking: recall@k, nDCG@k, MRR, MAP
  - Efficiency: parameters, size, inference speed
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import json
import logging
from collections import defaultdict

import numpy as np
import torch
from sentence_transformers import util

from src.SR.BI.BiEncoder import BiEncoder
from src.SR.BI.QADataset import QADataset


logger = logging.getLogger(__name__)

@dataclass
class EvaluationMetrics:
    """Evaluation results for a single model."""
    model_key: str
    num_examples: int

    # Similarity metrics
    mean_similarity: float
    std_similarity: float
    min_similarity: float
    max_similarity: float
    median_similarity: float

    # Ranking metrics
    recall_at_1: float = 0.0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    ndcg_at_5: float = 0.0
    ndcg_at_10: float = 0.0
    mrr: float = 0.0
    map_at_5: float = 0.0

    # Model metrics
    num_parameters: int = 0
    model_size_mb: float = 0.0
    inference_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict."""
        return {
            "model_key": self.model_key,
            "num_examples": self.num_examples,
            "mean_similarity": round(self.mean_similarity, 4),
            "std_similarity": round(self.std_similarity, 4),
            "min_similarity": round(self.min_similarity, 4),
            "max_similarity": round(self.max_similarity, 4),
            "median_similarity": round(self.median_similarity, 4),
            "recall_at_1": round(self.recall_at_1, 4),
            "recall_at_5": round(self.recall_at_5, 4),
            "recall_at_10": round(self.recall_at_10, 4),
            "ndcg_at_5": round(self.ndcg_at_5, 4),
            "ndcg_at_10": round(self.ndcg_at_10, 4),
            "mrr": round(self.mrr, 4),
            "map_at_5": round(self.map_at_5, 4),
            "num_parameters": self.num_parameters,
            "model_size_mb": round(self.model_size_mb, 2),
            "inference_time_ms": round(self.inference_time_ms, 2),
        }

class RetrievalMetrics:
    """Compute ranking metrics."""

    @staticmethod
    def recall_at_k(ranks: np.ndarray, k: int = 5) -> float:
        """Recall@k: fraction of queries with relevant doc in top-k."""
        return float(np.mean(ranks < k))

    @staticmethod
    def ndcg_at_k(relevances: np.ndarray, k: int = 5) -> float:
        """nDCG@k: Normalized Discounted Cumulative Gain."""
        relevances = relevances[:k]
        if len(relevances) == 0:
            return 0.0

        dcg = np.sum(relevances / np.log2(np.arange(2, len(relevances) + 2)))
        idcg = np.sum(np.ones(min(len(relevances), k)) / np.log2(np.arange(2, len(relevances) + 2)))

        return float(dcg / idcg) if idcg > 0 else 0.0

    @staticmethod
    def mrr(ranks: np.ndarray) -> float:
        """Mean Reciprocal Rank."""
        return float(np.mean(1.0 / (ranks + 1)))

    @staticmethod
    def map_at_k(ranks: np.ndarray, k: int = 5) -> float:
        """Mean Average Precision@k."""
        ranks = ranks[ranks < k]
        if len(ranks) == 0:
            return 0.0

        precisions = np.arange(1, len(ranks) + 1) / (ranks + 1)
        return float(np.mean(precisions))

class BiEncoderEvaluator:
    """
    Evaluate a trained BiEncoder on test data.

    Computes 18 metrics for comprehensive model comparison.
    """

    def __init__(
        self,
        model_key: str,
        model_dir: Path,
        test_jsonl: Path,
        max_examples: Optional[int] = None,
        device: Optional[str] = None,
    ) -> None:
        """Initialize evaluator."""
        self.model_key = model_key
        self.model_dir = Path(model_dir)
        self.test_jsonl = Path(test_jsonl)
        self.max_examples = max_examples

        logger.info(f"Loading model from {model_dir}")
        self.encoder = BiEncoder.load(str(model_dir), device=device)
        self.model = self.encoder.model

        logger.info(f"Loading test dataset from {test_jsonl}")
        self.test_dataset = QADataset(test_jsonl, max_examples=max_examples)

        self.metrics: Optional[EvaluationMetrics] = None

    def evaluate_similarity(self) -> Tuple[List[float], EvaluationMetrics]:
        """Evaluate cosine similarity between questions and contexts."""
        logger.info(f"Evaluating {self.model_key} on {len(self.test_dataset)} examples...")

        similarities = []
        for i, example in enumerate(self.test_dataset):
            if i % 50 == 0:
                logger.debug(f"  Processing example {i}/{len(self.test_dataset)}")

            q_emb = self.model.encode(example.question, convert_to_tensor=True)
            c_emb = self.model.encode(example.context, convert_to_tensor=True)

            sim = util.pytorch_cos_sim(q_emb, c_emb).item()
            similarities.append(sim)

        similarities = np.array(similarities)

        metrics = EvaluationMetrics(
            model_key=self.model_key,
            num_examples=len(self.test_dataset),
            mean_similarity=float(np.mean(similarities)),
            std_similarity=float(np.std(similarities)),
            min_similarity=float(np.min(similarities)),
            max_similarity=float(np.max(similarities)),
            median_similarity=float(np.median(similarities)),
        )

        logger.info(f"  Mean similarity: {metrics.mean_similarity:.4f}")
        logger.info(f"  Std similarity: {metrics.std_similarity:.4f}")

        return similarities, metrics

    def evaluate_ranking(
        self,
        corpus: Optional[List[str]] = None,
        top_k: int = 10,
    ) -> EvaluationMetrics:
        """Evaluate ranking metrics."""
        if self.metrics is None:
            _, self.metrics = self.evaluate_similarity()

        if corpus is None:
            corpus = [ex.context for ex in self.test_dataset]

        logger.info(f"Evaluating ranking metrics (corpus size={len(corpus)})...")

        # Encode corpus
        logger.debug("Encoding corpus...")
        corpus_embeddings = self.model.encode(
            corpus,
            batch_size=32,
            convert_to_tensor=True,
            show_progress_bar=True,
        )

        recalls_1, recalls_5, recalls_10 = [], [], []
        ndcgs_5, ndcgs_10 = [], []
        mrrs = []
        maps_5 = []

        for example in self.test_dataset:
            q_emb = self.model.encode(example.question, convert_to_tensor=True)

            # Compute similarities with all corpus documents
            sims = util.pytorch_cos_sim(q_emb, corpus_embeddings)[0].cpu().numpy()

            # Get ranking (higher similarity = better rank)
            ranked_idx = np.argsort(-sims)

            # Compute metrics
            relevances = np.array([1.0 if corpus[i] == example.context else 0.0
                                   for i in ranked_idx[:top_k]])

            ranks = np.where(relevances > 0)[0]
            if len(ranks) > 0:
                rank = ranks[0]
            else:
                rank = top_k

            recalls_1.append(float(rank < 1))
            recalls_5.append(float(rank < 5))
            recalls_10.append(float(rank < 10))

            ndcgs_5.append(RetrievalMetrics.ndcg_at_k(relevances, k=5))
            ndcgs_10.append(RetrievalMetrics.ndcg_at_k(relevances, k=10))
            mrrs.append(1.0 / (rank + 1))
            maps_5.append(RetrievalMetrics.map_at_k(relevances, k=5))

        self.metrics.recall_at_1 = float(np.mean(recalls_1))
        self.metrics.recall_at_5 = float(np.mean(recalls_5))
        self.metrics.recall_at_10 = float(np.mean(recalls_10))
        self.metrics.ndcg_at_5 = float(np.mean(ndcgs_5))
        self.metrics.ndcg_at_10 = float(np.mean(ndcgs_10))
        self.metrics.mrr = float(np.mean(mrrs))
        self.metrics.map_at_5 = float(np.mean(maps_5))

        logger.info(f"  Recall@5: {self.metrics.recall_at_5:.4f}")
        logger.info(f"  nDCG@5: {self.metrics.ndcg_at_5:.4f}")
        logger.info(f"  MRR: {self.metrics.mrr:.4f}")

        return self.metrics

    def evaluate_model_metrics(self) -> EvaluationMetrics:
        """Evaluate model size and inference speed."""
        if self.metrics is None:
            _, self.metrics = self.evaluate_similarity()

        logger.info("Evaluating model efficiency...")

        # Count parameters
        num_params = sum(p.numel() for p in self.encoder.model.parameters())
        self.metrics.num_parameters = num_params

        # Estimate model size
        model_size_mb = num_params * 4 / (1024 * 1024)
        self.metrics.model_size_mb = model_size_mb

        # Inference speed
        import time
        test_sents = [f"Test sentence {i}" for i in range(100)]
        start = time.time()
        _ = self.model.encode(test_sents, batch_size=32, show_progress_bar=False)
        inference_time = (time.time() - start) / len(test_sents) * 1000
        self.metrics.inference_time_ms = inference_time

        logger.info(f"  Parameters: {num_params:,}")
        logger.info(f"  Size: {model_size_mb:.1f} MB")
        logger.info(f"  Inference: {inference_time:.2f} ms/sentence")

        return self.metrics

    def evaluate_all(self) -> EvaluationMetrics:
        """Run all evaluations."""
        logger.info("=" * 80)
        logger.info(f"EVALUATING: {self.model_key}")
        logger.info("=" * 80)

        _, self.metrics = self.evaluate_similarity()
        self.evaluate_ranking()
        self.evaluate_model_metrics()

        return self.metrics