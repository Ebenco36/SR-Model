#!/usr/bin/env python3

"""
BiEncoder wrapper for HuggingFace models via sentence-transformers.

Supports:
  - bert-base-cased
  - allenai/scibert_scivocab_uncased
  - microsoft/BiomedNLP-PubMedBERT-base-uncased-abstracts
  - dmis-lab/biobert-base-cased-v1.1
  - microsoft/deberta-v3-base
"""

from typing import Optional, List, Union
import logging

import torch
from sentence_transformers import SentenceTransformer, models

from src.SR.BI.ModelConfig import get_model_config



logger = logging.getLogger(__name__)

def get_device() -> str:
    """Auto-detect best available device."""
    # Try CUDA first
    if torch.cuda.is_available():
        logger.info("✓ CUDA available")
        return "cuda"
    
    # Try MPS (Apple Silicon)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        logger.info("✓ MPS (Apple Silicon) available")
        return "mps"
    
    # Fallback to CPU
    logger.info("⚠ Using CPU (no GPU detected)")
    return "cpu"

class BiEncoder:
    """
    Wraps a HuggingFace encoder as a sentence-transformers bi-encoder.

    Supports all 5 models from bi_encoder_config.
    Supports CUDA, MPS (Apple Silicon), and CPU devices.
    """

    def __init__(
        self,
        model_key: str,
        device: Optional[str] = None,
    ) -> None:
        """
        Initialize BiEncoder.
        
        Args:
            model_key: Key from MODEL_REGISTRY
            device: Device to use ('cuda', 'mps', 'cpu'). Auto-detected if None.
        """
        self.config = get_model_config(model_key)
        self.model_key = model_key

        if device is None:
            device = get_device()
        
        self.device = device

        logger.info(f"Building BiEncoder: {model_key}")
        logger.info(f"  HF name: {self.config.hf_name}")
        logger.info(f"  Device: {self.device}")

        self._build_model()

    def _build_model(self) -> None:
        """Build SentenceTransformer from HF encoder + pooling."""
        hf_name = self.config.hf_name
        max_seq = self.config.max_seq_length
        pooling_mode = self.config.pooling

        logger.info(f"Loading transformer: {hf_name}")
        
        # Load transformer (without cache_folder parameter for compatibility)
        try:
            word_embedding_model = models.Transformer(
                hf_name,
                max_seq_length=max_seq,
            )
        except Exception as e:
            logger.error(f"Failed to load transformer: {e}")
            raise

        # Add pooling
        embedding_dim = word_embedding_model.get_word_embedding_dimension()

        if pooling_mode == "cls":
            pooling = models.Pooling(
                embedding_dim,
                pooling_mode_cls_token=True,
                pooling_mode_mean_tokens=False,
                pooling_mode_max_tokens=False,
            )
        elif pooling_mode == "mean":
            pooling = models.Pooling(
                embedding_dim,
                pooling_mode_cls_token=False,
                pooling_mode_mean_tokens=True,
                pooling_mode_max_tokens=False,
            )
        else:
            raise ValueError(f"Unknown pooling mode: {pooling_mode}")

        # Create SentenceTransformer
        logger.info("Creating SentenceTransformer with pooling")
        self.model = SentenceTransformer(
            modules=[word_embedding_model, pooling],
            device=self.device,
        )

        logger.info(f"✓ BiEncoder built successfully (dim={embedding_dim}, device={self.device})")

    def encode(
        self,
        sentences: Union[str, List[str]],
        batch_size: int = 32,
        convert_to_tensor: bool = False,
        show_progress_bar: bool = False,
        **kwargs
    ):
        """
        Encode sentences to embeddings.
        
        Args:
            sentences: String or list of strings to encode
            batch_size: Batch size for processing
            convert_to_tensor: Return torch tensor instead of numpy
            show_progress_bar: Show progress bar during encoding
            **kwargs: Additional arguments passed to model.encode()
        
        Returns:
            Numpy array or torch tensor of embeddings
        """
        return self.model.encode(
            sentences,
            batch_size=batch_size,
            convert_to_tensor=convert_to_tensor,
            show_progress_bar=show_progress_bar,
            **kwargs
        )

    def save(self, output_dir: str) -> None:
        """Save model to directory."""
        self.model.save(output_dir)
        logger.info(f"✓ Model saved to {output_dir}")

    @classmethod
    def load(cls, model_dir: str, device: Optional[str] = None) -> "BiEncoder":
        """
        Load a previously saved model.
        
        Args:
            model_dir: Directory containing saved model
            device: Device to load on. Auto-detected if None.
        
        Returns:
            BiEncoder instance with loaded model
        """
        if device is None:
            device = get_device()

        obj = object.__new__(cls)
        obj.model_key = "loaded"
        obj.config = None
        obj.device = device
        
        logger.info(f"Loading model from {model_dir}")
        obj.model = SentenceTransformer(model_dir, device=device)
        logger.info(f"✓ Model loaded successfully (device={device})")

        return obj

    def get_embedding_dimension(self) -> int:
        """Get embedding dimension."""
        return self.model.get_sentence_embedding_dimension()

    def __repr__(self) -> str:
        return (
            f"BiEncoder(model_key={self.model_key}, "
            f"device={self.device}, "
            f"dim={self.get_embedding_dimension()})"
        )