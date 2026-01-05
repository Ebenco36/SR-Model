#!/usr/bin/env python3

"""
BiEncoder Inference Engine - DYNAMIC MODEL SWITCHING.

Features:
  ✅ Load any trained model
  ✅ Switch between models dynamically
  ✅ Single/batch similarity
  ✅ Document retrieval (top-k)
  ✅ Semantic search with ranking
  ✅ Multi-corpus search
  ✅ Caching and batching
  ✅ List available trained models
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging
from collections import defaultdict

import numpy as np
import torch
from sentence_transformers import util, SentenceTransformer

from src.SR.BI.BiEncoder import get_device

logger = logging.getLogger(__name__)


@dataclass
class SimilarityResult:
    """Result of similarity computation."""
    query: str
    context: str
    score: float
    rank: Optional[int] = None
    model: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict."""
        return {
            "query": self.query,
            "context": self.context,
            "score": round(self.score, 4),
            "rank": self.rank,
            "model": self.model,
        }


@dataclass
class RetrievalResult:
    """Result of document retrieval."""
    rank: int
    context: str
    score: float
    doc_id: Optional[str] = None
    model: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict."""
        return {
            "rank": self.rank,
            "context": self.context,
            "score": round(self.score, 4),
            "doc_id": self.doc_id,
            "model": self.model,
        }


class BiEncoderInference:
    """
    Inference engine for trained bi-encoder models.

    ✅ Load any trained model
    ✅ Switch between models dynamically
    ✅ Single similarity computation
    ✅ Batch similarity computation
    ✅ Document retrieval (top-k)
    ✅ Semantic search with ranking
    ✅ Corpus caching for efficiency
    """

    def __init__(
        self,
        model_dir: Path,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        corpus: Optional[List[str]] = None,
        corpus_ids: Optional[List[str]] = None,
        cache_embeddings: bool = True,
    ):
        """
        Initialize inference engine.

        Args:
            model_dir: Path to trained model directory
            model_name: Name of model (for tracking)
            device: Device ('cuda', 'mps', 'cpu'). Auto-detected if None.
            corpus: Optional list of documents to encode
            corpus_ids: Optional document IDs (same length as corpus)
            cache_embeddings: Cache corpus embeddings for faster retrieval
        """
        self.model_dir = Path(model_dir)
        self.model_name = model_name or self.model_dir.name
        
        if device is None:
            device = get_device()
        self.device = device

        logger.info(f"Loading model: {self.model_name}")
        logger.info(f"  Path: {self.model_dir}")
        logger.info(f"  Device: {self.device}")

        try:
            self.model = SentenceTransformer(str(self.model_dir), device=self.device)
            logger.info(f"  Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

        # Corpus management
        self.corpus = corpus or []
        self.corpus_ids = corpus_ids or [f"doc_{i}" for i in range(len(self.corpus))]
        self.corpus_embeddings = None
        self.cache_enabled = cache_embeddings

        if self.corpus and cache_embeddings:
            self._cache_corpus()

    def _cache_corpus(self) -> None:
        """Pre-compute and cache corpus embeddings."""
        logger.info(f"Caching embeddings for {len(self.corpus)} documents...")
        self.corpus_embeddings = self.model.encode(
            self.corpus,
            batch_size=32,
            convert_to_tensor=True,
            show_progress_bar=True,
        )
        logger.info(f"✓ Cached {len(self.corpus)} embeddings")

    def set_corpus(
        self,
        corpus: List[str],
        corpus_ids: Optional[List[str]] = None,
    ) -> None:
        """
        Set corpus for retrieval.

        Args:
            corpus: List of documents
            corpus_ids: Optional document IDs
        """
        if len(corpus) == 0:
            raise ValueError("Corpus cannot be empty")

        self.corpus = corpus
        self.corpus_ids = corpus_ids or [f"doc_{i}" for i in range(len(corpus))]
        self.corpus_embeddings = None

        if self.cache_enabled:
            self._cache_corpus()

    def similarity(
        self,
        query: str,
        context: str,
    ) -> SimilarityResult:
        """
        Compute similarity between query and context.

        Args:
            query: Query text
            context: Context/document text

        Returns:
            SimilarityResult with score
        """
        q_emb = self.model.encode(query, convert_to_tensor=True)
        c_emb = self.model.encode(context, convert_to_tensor=True)

        score = util.pytorch_cos_sim(q_emb, c_emb).item()

        return SimilarityResult(
            query=query,
            context=context,
            score=score,
            model=self.model_name,
        )

    def batch_similarity(
        self,
        queries: List[str],
        contexts: List[str],
    ) -> List[SimilarityResult]:
        """
        Compute similarity for multiple query-context pairs.

        Args:
            queries: List of queries
            contexts: List of contexts (same length as queries)

        Returns:
            List of SimilarityResult
        """
        if len(queries) != len(contexts):
            raise ValueError("queries and contexts must have same length")

        logger.info(f"Computing similarity for {len(queries)} pairs...")

        q_embeddings = self.model.encode(
            queries,
            batch_size=32,
            convert_to_tensor=True,
            show_progress_bar=True,
        )

        c_embeddings = self.model.encode(
            contexts,
            batch_size=32,
            convert_to_tensor=True,
            show_progress_bar=True,
        )

        results = []
        for i in range(len(queries)):
            score = util.pytorch_cos_sim(
                q_embeddings[i:i+1],
                c_embeddings[i:i+1]
            ).item()

            results.append(SimilarityResult(
                query=queries[i],
                context=contexts[i],
                score=score,
                model=self.model_name,
            ))

        return results

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[RetrievalResult]:
        """
        Retrieve top-k documents for query.

        Args:
            query: Query text
            top_k: Number of documents to return

        Returns:
            List of RetrievalResult sorted by score
        """
        if not self.corpus:
            raise ValueError("No corpus set. Use set_corpus() first.")

        if self.corpus_embeddings is None:
            self._cache_corpus()

        logger.debug(f"[{self.model_name}] Retrieving top-{top_k} for: {query[:50]}...")

        q_emb = self.model.encode(query, convert_to_tensor=True)

        # Compute similarities
        sims = util.pytorch_cos_sim(q_emb, self.corpus_embeddings)[0].cpu().numpy()

        # Get top-k indices
        top_indices = np.argsort(-sims)[:top_k]

        results = []
        for rank, idx in enumerate(top_indices, 1):
            results.append(RetrievalResult(
                rank=rank,
                context=self.corpus[idx],
                score=float(sims[idx]),
                doc_id=self.corpus_ids[idx],
                model=self.model_name,
            ))

        return results

    def batch_retrieve(
        self,
        queries: List[str],
        top_k: int = 10,
    ) -> Dict[str, List[RetrievalResult]]:
        """
        Retrieve top-k documents for multiple queries.

        Args:
            queries: List of queries
            top_k: Number of documents per query

        Returns:
            Dict mapping query to list of RetrievalResult
        """
        if not self.corpus:
            raise ValueError("No corpus set. Use set_corpus() first.")

        if self.corpus_embeddings is None:
            self._cache_corpus()

        logger.info(f"[{self.model_name}] Retrieving top-{top_k} for {len(queries)} queries...")

        q_embeddings = self.model.encode(
            queries,
            batch_size=32,
            convert_to_tensor=True,
            show_progress_bar=True,
        )

        results = {}
        for i, query in enumerate(queries):
            # Compute similarities
            sims = util.pytorch_cos_sim(
                q_embeddings[i:i+1],
                self.corpus_embeddings
            )[0].cpu().numpy()

            # Get top-k indices
            top_indices = np.argsort(-sims)[:top_k]

            query_results = []
            for rank, idx in enumerate(top_indices, 1):
                query_results.append(RetrievalResult(
                    rank=rank,
                    context=self.corpus[idx],
                    score=float(sims[idx]),
                    doc_id=self.corpus_ids[idx],
                    model=self.model_name,
                ))

            results[query] = query_results

        return results

    def search(
        self,
        query: str,
        top_k: int = 10,
        threshold: float = 0.0,
    ) -> List[RetrievalResult]:
        """
        Semantic search with optional threshold filtering.

        Args:
            query: Query text
            top_k: Maximum number of results
            threshold: Minimum similarity score (0.0-1.0)

        Returns:
            Filtered retrieval results
        """
        results = self.retrieve(query, top_k=top_k)

        # Filter by threshold
        filtered = [r for r in results if r.score >= threshold]

        logger.info(f"[{self.model_name}] Search returned {len(filtered)}/{len(results)} results (threshold={threshold})")

        return filtered

    def rank(
        self,
        query: str,
        contexts: List[str],
    ) -> List[RetrievalResult]:
        """
        Rank a list of contexts for a query.

        Args:
            query: Query text
            contexts: List of contexts to rank

        Returns:
            Ranked contexts
        """
        logger.debug(f"[{self.model_name}] Ranking {len(contexts)} contexts...")

        q_emb = self.model.encode(query, convert_to_tensor=True)

        c_embeddings = self.model.encode(
            contexts,
            batch_size=32,
            convert_to_tensor=True,
        )

        sims = util.pytorch_cos_sim(q_emb, c_embeddings)[0].cpu().numpy()

        # Sort by score
        sorted_indices = np.argsort(-sims)

        results = []
        for rank, idx in enumerate(sorted_indices, 1):
            results.append(RetrievalResult(
                rank=rank,
                context=contexts[idx],
                score=float(sims[idx]),
                model=self.model_name,
            ))

        return results

    def __call__(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[RetrievalResult]:
        """Allow using model as function."""
        return self.retrieve(query, top_k=top_k)


class BiEncoderSearchEngine:
    """
    High-level search engine with dynamic model switching.

    Features:
      ✅ Load multiple trained models
      ✅ Switch between models dynamically
      ✅ Multi-corpus search
      ✅ Compare results across models
      ✅ Caching strategies
      ✅ Search analytics
    """

    def __init__(
        self,
        models_dir: Path = Path("results/models"),
        device: Optional[str] = None,
    ):
        """
        Initialize search engine.

        Args:
            models_dir: Directory containing trained models
            device: Device to use
        """
        self.models_dir = Path(models_dir)
        self.device = device or get_device()

        self.models: Dict[str, BiEncoderInference] = {}
        self.corpora: Dict[str, List[str]] = {}
        self.corpus_ids: Dict[str, List[str]] = {}
        self.search_stats = defaultdict(int)

        logger.info(f"Search engine initialized")
        logger.info(f"  Models dir: {self.models_dir}")
        logger.info(f"  Device: {self.device}")

    def load_model(
        self,
        model_name: str,
        model_path: Optional[Path] = None,
    ) -> None:
        """
        Load a trained model.

        Args:
            model_name: Name to reference model
            model_path: Path to model (defaults to models_dir/model_name)
        """
        if model_path is None:
            model_path = self.models_dir / model_name

        if model_path not in self.models_dir.parent.glob("**/*") and not model_path.exists():
            logger.error(f"Model path not found: {model_path}")
            raise FileNotFoundError(f"Model not found: {model_path}")

        try:
            self.models[model_name] = BiEncoderInference(
                model_dir=model_path,
                model_name=model_name,
                device=self.device,
            )
            logger.info(f"✓ Loaded model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise

    def load_all_models(self) -> List[str]:
        """
        Auto-load all trained models from models_dir.

        Returns:
            List of loaded model names
        """
        loaded = []
        
        if not self.models_dir.exists():
            logger.warning(f"Models directory not found: {self.models_dir}")
            return loaded

        for model_path in self.models_dir.iterdir():
            if model_path.is_dir() and (model_path / "pytorch_model.bin").exists():
                try:
                    self.load_model(model_path.name, model_path)
                    loaded.append(model_path.name)
                except Exception as e:
                    logger.warning(f"Could not load {model_path.name}: {e}")

        logger.info(f"✓ Loaded {len(loaded)} models: {loaded}")
        return loaded

    def add_corpus(
        self,
        corpus_name: str,
        documents: List[str],
        doc_ids: Optional[List[str]] = None,
    ) -> None:
        """
        Add a corpus for searching.

        Args:
            corpus_name: Name of corpus
            documents: List of documents
            doc_ids: Optional document IDs
        """
        logger.info(f"Adding corpus '{corpus_name}' with {len(documents)} documents")

        self.corpora[corpus_name] = documents
        self.corpus_ids[corpus_name] = doc_ids or [f"{corpus_name}_doc_{i}" for i in range(len(documents))]

        # Set corpus for all loaded models
        for model_name, model in self.models.items():
            model.set_corpus(documents, self.corpus_ids[corpus_name])

    def search_model(
        self,
        model_name: str,
        query: str,
        top_k: int = 10,
    ) -> List[RetrievalResult]:
        """
        Search using a specific model.

        Args:
            model_name: Name of model to use
            query: Query text
            top_k: Number of results

        Returns:
            Retrieval results
        """
        if model_name not in self.models:
            raise ValueError(f"Model not loaded: {model_name}")

        self.search_stats[model_name] += 1
        return self.models[model_name].retrieve(query, top_k=top_k)

    def search_all_models(
        self,
        query: str,
        top_k: int = 10,
    ) -> Dict[str, List[RetrievalResult]]:
        """
        Search using all loaded models and compare results.

        Args:
            query: Query text
            top_k: Number of results per model

        Returns:
            Dict mapping model name to results
        """
        logger.info(f"Searching with {len(self.models)} models: {query[:50]}...")

        results = {}
        for model_name in self.models:
            results[model_name] = self.search_model(model_name, query, top_k)

        return results

    def compare_models(
        self,
        query: str,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Compare results across all models.

        Args:
            query: Query text
            top_k: Number of results per model

        Returns:
            Comparison data
        """
        results = self.search_all_models(query, top_k)

        # Build comparison
        comparison = {
            "query": query,
            "models": {},
        }

        for model_name, model_results in results.items():
            comparison["models"][model_name] = {
                "results": [r.to_dict() for r in model_results],
                "avg_score": np.mean([r.score for r in model_results]),
                "max_score": max([r.score for r in model_results]),
                "min_score": min([r.score for r in model_results]),
            }

        return comparison

    def get_stats(self) -> Dict[str, int]:
        """Get search statistics by model."""
        return dict(self.search_stats)

    def list_models(self) -> List[str]:
        """List all loaded models."""
        return list(self.models.keys())