#!/usr/bin/env python3

"""
BiEncoder Trainer for fine-tuning on QA datasets.

Trains any HF encoder model as a bi-encoder on QA pairs 
using MultipleNegativesRankingLoss.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import json
import logging
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from sentence_transformers import losses
from sentence_transformers.evaluation import InformationRetrievalEvaluator

from src.SR.BI.BiEncoder import BiEncoder, get_device
from src.SR.BI.QADataset import QADataset


logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Training configuration."""
    batch_size: int = 32
    epochs: int = 3
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "warmup_ratio": self.warmup_ratio,
            "weight_decay": self.weight_decay,
            "max_grad_norm": self.max_grad_norm,
            "seed": self.seed,
        }

class BiEncoderTrainer:
    """
    Train BiEncoder models.
    ✅ Saves best model based on validation IR metrics
    ✅ Supports optional validation set
    ✅ Production-grade error handling
    """

    def __init__(
        self,
        model_key: str,
        train_jsonl: Path,
        output_dir: Path = Path("models"),
        val_jsonl: Optional[Path] = None,
        config: TrainingConfig = TrainingConfig(),
        device: Optional[str] = None,
    ) -> None:
        """Initialize trainer."""
        self.model_key = model_key
        self.train_jsonl = Path(train_jsonl)
        self.val_jsonl = Path(val_jsonl) if val_jsonl else None
        self.output_dir = Path(output_dir) / model_key
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config

        if device is None:
            device = get_device()
        self.device = device

        torch.manual_seed(config.seed)

        logger.info(f"Initializing BiEncoderTrainer")
        logger.info(f"  Model: {model_key}")
        logger.info(f"  Device: {self.device}")

        # Initialize model
        self.encoder = BiEncoder(model_key, device=self.device)
        self.model = self.encoder.model

        # Load training data
        logger.info("Loading datasets...")
        self.train_dataset = QADataset(self.train_jsonl)
        
        # Setup validation evaluator (CORRECT USAGE)
        self.evaluator = None
        self.has_validation = False
        
        if self.val_jsonl and self.val_jsonl.exists():
            logger.info(f"Setting up validation evaluator...")
            self.val_dataset = QADataset(self.val_jsonl)
            self.evaluator = self._build_evaluator(self.val_dataset)
            self.has_validation = True
            logger.info("✓ Validation evaluator ready")
        else:
            logger.warning("⚠ No validation set provided - saving last model only")

        self.train_history: Dict[str, Any] = {}

    def _build_evaluator(self, dataset: QADataset) -> InformationRetrievalEvaluator:
        """
        Build evaluator from QADataset.
        
        Correct format:
        - queries: Dict[id, text]
        - corpus: Dict[id, text]
        - relevant_docs: Dict[query_id, Set[doc_id]]
        """
        queries = {}
        corpus = {}
        relevant_docs = {}

        # Iterate through dataset
        for example in dataset:
            # Store query
            queries[example.id] = example.question
            
            # Store document in corpus
            corpus[example.doc_id] = example.context
            
            # Track relevance (which docs are relevant to which queries)
            if example.id not in relevant_docs:
                relevant_docs[example.id] = set()
            relevant_docs[example.id].add(example.doc_id)

        logger.info(f"  Evaluator: {len(queries)} queries, {len(corpus)} docs")

        # Create evaluator (NO main_score_function parameter)
        return InformationRetrievalEvaluator(
            queries=queries,
            corpus=corpus,
            relevant_docs=relevant_docs,
            name=f"val_{self.model_key}",
            show_progress_bar=False,
        )

    def _build_train_loader(self) -> DataLoader:
        """Create training DataLoader."""
        examples = self.train_dataset.to_input_examples()
        return DataLoader(
            examples,
            shuffle=True,
            batch_size=self.config.batch_size,
        )

    def _build_loss(self):
        """Create loss function."""
        return losses.MultipleNegativesRankingLoss(self.model)

    def train(self) -> Dict[str, Any]:
        """Train the model."""
        logger.info("=" * 80)
        logger.info(f"TRAINING: {self.model_key}")
        logger.info("=" * 80)

        train_loader = self._build_train_loader()
        train_loss = self._build_loss()

        total_steps = len(train_loader) * self.config.epochs
        warmup_steps = max(1, int(self.config.warmup_ratio * total_steps))

        start_time = datetime.now()

        # Only save best model if we have validation
        save_best = self.has_validation
        
        if save_best:
            logger.info(f"✓ Validation active: Will save BEST model")
        else:
            logger.info(f"⚠ No validation: Saving LAST model only")

        try:
            self.model.fit(
                train_objectives=[(train_loader, train_loss)],
                epochs=self.config.epochs,
                warmup_steps=warmup_steps,
                optimizer_params={
                    "lr": self.config.learning_rate,
                    "weight_decay": self.config.weight_decay,
                    "eps": 1e-6,
                },
                output_path=str(self.output_dir),
                show_progress_bar=True,
                save_best_model=save_best,
                evaluator=self.evaluator if save_best else None,
                evaluation_steps=0,  # Evaluate at end of each epoch
                checkpoint_save_steps=0,  # Don't save intermediate checkpoints
                checkpoint_save_total_limit=1,  # Keep only best model
            )

            training_time = (datetime.now() - start_time).total_seconds()
            
            stats = {
                "model_key": self.model_key,
                "total_steps": total_steps,
                "training_time_seconds": training_time,
                "best_model_saved": save_best,
                "config": self.config.to_dict(),
            }

            self.train_history = stats
            self.save_config()
            
            logger.info(f"✓ Training completed in {training_time:.1f}s")
            return stats

        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise

    def save_config(self) -> None:
        """Save training config."""
        config_file = self.output_dir / "training_config.json"
        with open(config_file, "w") as f:
            json.dump(self.train_history, f, indent=2)

    def get_model(self) -> BiEncoder:
        """Return the trained encoder."""
        return self.encoder