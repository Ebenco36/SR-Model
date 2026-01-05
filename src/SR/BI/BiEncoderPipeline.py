#!/usr/bin/env python3

"""
Complete end-to-end pipeline for training and evaluating BiEncoder models.

Usage:
    python main_pipeline.py \
        --qa-train qa_dataset/train/train.jsonl \
        --qa-val qa_dataset/validation/validation.jsonl \
        --qa-test qa_dataset/test/test.jsonl \
        --models bert-base-cased scibert pubmedbert biobert deberta-v3-base \
        --output-dir results
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

from src.SR.BI.BiEncoder import get_device
from src.SR.BI.BiEncoderTrainer import BiEncoderTrainer, TrainingConfig
from src.SR.BI.BiEncoderVisualizer import BiEncoderVisualizer
from src.SR.BI.EvaluationMetrics import BiEncoderEvaluator
from src.SR.BI.ModelConfig import MODEL_REGISTRY, list_available_models



# Setup logging to both stdout and file
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("pipeline.log"),
    ],
)
logger = logging.getLogger(__name__)


def list_available_models() -> Dict[str, str]:
    """Get all available models from registry."""
    return MODEL_REGISTRY


class BiEncoderPipeline:
    """
    End-to-end training and evaluation pipeline.
    
    ✅ Train multiple models
    ✅ Evaluate on test set
    ✅ Generate visualizations
    ✅ Save comprehensive summary
    """

    def __init__(
        self,
        qa_train: Path,
        qa_test: Path,
        output_dir: Path = Path("results"),
        qa_val: Optional[Path] = None,
        config: TrainingConfig = TrainingConfig(),
    ) -> None:
        """
        Initialize pipeline.

        Args:
            qa_train: Training JSONL path
            qa_test: Test JSONL path
            output_dir: Output directory
            qa_val: Optional validation JSONL
            config: Training config
        """
        self.qa_train = Path(qa_train)
        self.qa_val = Path(qa_val) if qa_val else None
        self.qa_test = Path(qa_test)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config

        self.train_results: Dict[str, Dict[str, Any]] = {}
        self.eval_results: Dict[str, Dict[str, Any]] = {}
        self.trained_models: Dict[str, str] = {}  # model_key -> model_dir

        logger.info(f"Pipeline initialized")
        logger.info(f"  Output: {self.output_dir}")
        logger.info(f"  Train: {self.qa_train}")
        logger.info(f"  Test: {self.qa_test}")
        if self.qa_val:
            logger.info(f"  Val: {self.qa_val}")

    def train_models(self, model_keys: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Train all specified models.

        Args:
            model_keys: List of model keys

        Returns:
            Training results dict
        """
        logger.info("=" * 80)
        logger.info("TRAINING PHASE")
        logger.info("=" * 80)

        for model_key in model_keys:
            if model_key not in MODEL_REGISTRY:
                logger.error(f"Unknown model: {model_key}")
                continue

            logger.info(f"\n{'=' * 80}")
            logger.info(f"TRAINING: {model_key}")
            logger.info(f"{'=' * 80}")

            try:
                trainer = BiEncoderTrainer(
                    model_key=model_key,
                    train_jsonl=self.qa_train,
                    val_jsonl=self.qa_val,
                    output_dir=self.output_dir / "models",
                    config=self.config,
                )

                stats = trainer.train()
                model_dir = self.output_dir / "models" / model_key
                
                self.train_results[model_key] = stats
                self.trained_models[model_key] = str(model_dir)
                
                logger.info(f"✓ {model_key} training completed")
                logger.info(f"  Model saved to: {model_dir}")

            except Exception as e:
                logger.error(f"✗ {model_key} training failed: {e}")
                logger.exception(e)

        logger.info("")
        logger.info(f"Successfully trained: {list(self.trained_models.keys())}")
        return self.train_results

    def evaluate_models(self, model_keys: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate all trained models.

        Args:
            model_keys: List of model keys

        Returns:
            Evaluation results dict
        """
        logger.info("=" * 80)
        logger.info("EVALUATION PHASE")
        logger.info("=" * 80)

        for model_key in model_keys:
            if model_key not in self.trained_models:
                logger.warning(f"Skipping {model_key} (not trained)")
                continue

            logger.info(f"\n{'=' * 80}")
            logger.info(f"EVALUATING: {model_key}")
            logger.info(f"{'=' * 80}")

            try:
                model_dir = Path(self.trained_models[model_key])

                evaluator = BiEncoderEvaluator(
                    model_dir=model_dir,
                )

                metrics = evaluator.evaluate(self.qa_test)
                self.eval_results[model_key] = metrics
                
                logger.info(f"✓ {model_key} evaluation completed")
                logger.info(f"  Metrics: {metrics}")

            except Exception as e:
                logger.error(f"✗ {model_key} evaluation failed: {e}")
                logger.exception(e)

        logger.info("")
        logger.info(f"Evaluated: {list(self.eval_results.keys())}")
        return self.eval_results

    def visualize_results(self) -> List[str]:
        """
        Generate visualizations.

        Returns:
            List of output file paths
        """
        logger.info("=" * 80)
        logger.info("VISUALIZATION PHASE")
        logger.info("=" * 80)

        if not self.eval_results:
            logger.warning("No evaluation results to visualize")
            return []

        try:
            viz_dir = self.output_dir / "visualizations"
            visualizer = BiEncoderVisualizer(output_dir=viz_dir)

            for model_key, metrics in self.eval_results.items():
                visualizer.add_result(model_key, metrics)

            outputs = visualizer.generate_all()
            visualizer.save_results_json()
            visualizer.save_results_csv()
            visualizer.generate_html_index()

            logger.info("")
            logger.info("✓ Visualizations generated successfully")
            for output in outputs:
                logger.info(f"  ✓ {output}")

            return outputs

        except Exception as e:
            logger.error(f"Visualization failed: {e}")
            logger.exception(e)
            return []

    def save_summary(self) -> str:
        """
        Save pipeline summary.

        Returns:
            Path to summary file
        """
        summary = {
            "timestamp": datetime.now().isoformat(),
            "device": get_device(),
            "config": self.config.to_dict(),
            "training_results": self.train_results,
            "evaluation_results": self.eval_results,
            "successfully_trained": list(self.trained_models.keys()),
            "successfully_evaluated": list(self.eval_results.keys()),
        }

        summary_file = self.output_dir / "summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"✓ Summary saved to {summary_file}")
        return str(summary_file)

    def run(self, model_keys: List[str]) -> None:
        """
        Run complete pipeline.

        Args:
            model_keys: List of model keys to train/evaluate
        """
        logger.info("=" * 80)
        logger.info("BiEncoder Training & Evaluation Pipeline")
        logger.info("=" * 80)
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Models to train: {model_keys}")
        logger.info(f"Device: {get_device()}")
        logger.info("=" * 80)

        # Train
        self.train_models(model_keys)

        # Evaluate (only trained models)
        self.evaluate_models(list(self.trained_models.keys()))

        # Visualize
        self.visualize_results()

        # Summary
        self.save_summary()

        logger.info("")
        logger.info("=" * 80)
        if self.trained_models:
            logger.info("✓ PIPELINE COMPLETED SUCCESSFULLY")
        else:
            logger.warning("⚠ No models trained successfully")
        logger.info("=" * 80)
        logger.info(f"Results saved to: {self.output_dir}")
        logger.info(f"Summary: {self.output_dir}/summary.json")
        logger.info(f"Visualizations: {self.output_dir}/visualizations/index.html")
        logger.info(f"Pipeline log: pipeline.log")


def main():
    parser = argparse.ArgumentParser(
        description="BiEncoder Training & Evaluation Pipeline"
    )

    parser.add_argument("--qa-train", required=True, help="Training JSONL")
    parser.add_argument("--qa-val", help="Validation JSONL (optional)")
    parser.add_argument("--qa-test", required=True, help="Test JSONL")
    parser.add_argument(
        "--models",
        nargs="+",
        help="Models to train (default: all)",
        choices=list(MODEL_REGISTRY.keys()),
    )
    parser.add_argument("--output-dir", default="results", help="Output directory")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--list-models", action="store_true", help="List available models"
    )

    args = parser.parse_args()

    # List models
    if args.list_models:
        logger.info("Available models:")
        for key, value in list_available_models().items():
            logger.info(f"  {key:<25} - {value}")
        return

    # Validate inputs
    if not Path(args.qa_train).exists():
        logger.error(f"Training file not found: {args.qa_train}")
        sys.exit(1)

    if not Path(args.qa_test).exists():
        logger.error(f"Test file not found: {args.qa_test}")
        sys.exit(1)

    if args.qa_val and not Path(args.qa_val).exists():
        logger.error(f"Validation file not found: {args.qa_val}")
        sys.exit(1)

    # Config
    config = TrainingConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )

    # Run pipeline
    pipeline = BiEncoderPipeline(
        qa_train=Path(args.qa_train),
        qa_val=Path(args.qa_val) if args.qa_val else None,
        qa_test=Path(args.qa_test),
        output_dir=Path(args.output_dir),
        config=config,
    )

    models = args.models if args.models else list(MODEL_REGISTRY.keys())
    pipeline.run(models)


if __name__ == "__main__":
    main()