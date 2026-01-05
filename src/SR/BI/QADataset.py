#!/usr/bin/env python3

"""
QA Dataset loader for SQuAD-style data from qa_generator.py
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Iterator, Optional
from pathlib import Path
import json
import logging

from sentence_transformers import InputExample

logger = logging.getLogger(__name__)

@dataclass
class QAPair:
    """Single QA pair from qa_generator output."""
    id: str
    question: str
    context: str
    answers: Dict[str, Any]
    doc_id: str

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "QAPair":
        """Create from JSON dict."""
        return cls(
            id=d.get("id", ""),
            question=d.get("question", ""),
            context=d.get("context", ""),
            answers=d.get("answers", {}),
            doc_id=d.get("doc_id", ""),
        )

class QADataset:
    """
    Loader for QA JSONL data from qa_generator.

    Each line is a JSON object with:
      {
        "id": "...",
        "doc_id": "...",
        "question": "...",
        "context": "...",
        "answers": {"text": [...], "answer_start": [...]},
        "metadata": {...}
      }
    """

    def __init__(self, jsonl_path: Path, max_examples: Optional[int] = None):
        self.path = Path(jsonl_path)
        self.max_examples = max_examples
        self.examples: List[QAPair] = []
        self._load()

    def _load(self) -> None:
        """Load examples from JSONL file."""
        if not self.path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.path}")

        with self.path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if not line.strip():
                    continue
                try:
                    d = json.loads(line)
                    example = QAPair.from_dict(d)
                    if example.question and example.context:
                        self.examples.append(example)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping malformed JSON at line {i}: {e}")
                    continue

                if self.max_examples and len(self.examples) >= self.max_examples:
                    break

        logger.info(f"Loaded {len(self.examples)} QA examples from {self.path}")

    def to_input_examples(self, max_pairs: Optional[int] = None) -> List[InputExample]:
        """
        Convert QA pairs to InputExample for sentence-transformers.

        For MultipleNegativesRankingLoss:
          - texts[0] = question (query)
          - texts[1] = context (positive document)
          - in-batch negatives are sampled automatically

        Args:
            max_pairs: Limit number of pairs (for testing)

        Returns:
            List of InputExample objects
        """
        examples = self.examples
        if max_pairs:
            examples = examples[:max_pairs]

        data = []
        for ex in examples:
            data.append(InputExample(texts=[ex.question, ex.context]))

        return data

    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        if not self.examples:
            return {"num_examples": 0}

        question_lengths = [len(ex.question.split()) for ex in self.examples]
        context_lengths = [len(ex.context.split()) for ex in self.examples]

        return {
            "num_examples": len(self.examples),
            "num_docs": len(set(ex.doc_id for ex in self.examples)),
            "avg_question_length": sum(question_lengths) / len(question_lengths),
            "avg_context_length": sum(context_lengths) / len(context_lengths),
            "min_question_length": min(question_lengths),
            "max_question_length": max(question_lengths),
            "min_context_length": min(context_lengths),
            "max_context_length": max(context_lengths),
        }

    def __len__(self) -> int:
        return len(self.examples)

    def __iter__(self) -> Iterator[QAPair]:
        yield from self.examples

    def __getitem__(self, idx: int) -> QAPair:
        return self.examples[idx]