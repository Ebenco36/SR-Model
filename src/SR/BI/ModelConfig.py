#!/usr/bin/env python3

"""
Configuration for BiEncoder models.

Supports:
  - bert-base-cased
  - allenai/scibert_scivocab_uncased
  - microsoft/BiomedNLP-PubMedBERT-base-uncased-abstracts
  - dmis-lab/biobert-base-cased-v1.1
  - microsoft/deberta-v3-base
"""

from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class ModelConfig:
    """Configuration for a single model."""
    key: str
    hf_name: str
    pooling: str = "cls"
    max_seq_length: int = 512
    category: str = "general"  # general, scientific, biomedical, clinical
    description: str = ""

# Global registry
MODEL_REGISTRY: Dict[str, ModelConfig] = {
    "bert-base-cased": ModelConfig(
        key="bert-base-cased",
        hf_name="bert-base-cased",
        category="general",
        description="BERT base cased (general-purpose baseline)"
    ),
    "scibert": ModelConfig(
        key="scibert",
        hf_name="allenai/scibert_scivocab_uncased",
        category="scientific",
        description="SciBERT (pre-trained on scientific papers)"
    ),
    "pubmedbert": ModelConfig(
        key="pubmedbert",
        hf_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstracts",
        category="clinical",
        description="PubMedBERT (pre-trained on PubMed abstracts)"
    ),
    "biobert": ModelConfig(
        key="biobert",
        hf_name="dmis-lab/biobert-base-cased-v1.1",
        category="biomedical",
        description="BioBERT (pre-trained on biomedical text)"
    ),
    "deberta-v3-base": ModelConfig(
        key="deberta-v3-base",
        hf_name="microsoft/deberta-v3-base",
        category="general",
        description="DeBERTa v3 base (improved decoder architecture)"
    ),
}

def get_model_config(model_key: str) -> ModelConfig:
    """Get configuration for a model by key."""
    if model_key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model_key: {model_key}. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[model_key]

def list_available_models() -> Dict[str, str]:
    """List all available models with descriptions."""
    return {k: v.description for k, v in MODEL_REGISTRY.items()}