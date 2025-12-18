from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from datasets import Dataset


def _get_first_answer(ex: Dict[str, Any]) -> Tuple[Optional[str], Optional[int]]:
    ans = ex.get("answers", {}) or {}
    texts = ans.get("text", []) or []
    starts = ans.get("answer_start", []) or []
    if not texts or not starts:
        return None, None
    if not isinstance(texts[0], str):
        return None, None
    if not isinstance(starts[0], int):
        return None, None
    return texts[0], starts[0]


@dataclass(frozen=True)
class QASlidingWindowFeaturizer:
    """
    Featurizes SQuAD-style examples with sliding window for long contexts.
    Produces HF-ready features including start_positions/end_positions.

    - return_overflowing_tokens=True
    - stride controls overlap between chunks
    - supports unanswerable examples (answers.text == [])
    """
    tokenizer: Any
    max_len: int = 384
    doc_stride: int = 128
    pad_on_right: bool = True  # for most BERT-like models, question on left

    def featurize(self, examples: List[Dict[str, Any]], is_train: bool = True) -> Dataset:
        features = []

        for ex in examples:
            q = ex["question"]
            c = ex["context"]
            ex_id = ex.get("id", "")
            doc_id = ex.get("doc_id", "")

            answer_text, answer_start = _get_first_answer(ex)
            answer_end = (answer_start + len(answer_text)) if (answer_text is not None and answer_start is not None) else None

            # Tokenize with overflow and offsets
            tokenized = self.tokenizer(
                q if self.pad_on_right else c,
                c if self.pad_on_right else q,
                truncation="only_second" if self.pad_on_right else "only_first",
                max_length=self.max_len,
                stride=self.doc_stride,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding="max_length",
            )

            overflow_to_sample = tokenized.pop("overflow_to_sample_mapping", None)
            # Note: in our case each call is for one sample, so overflow map is 0..n-1
            offset_mapping = tokenized["offset_mapping"]

            for i in range(len(tokenized["input_ids"])):
                input_ids = tokenized["input_ids"][i]
                attention_mask = tokenized["attention_mask"][i]
                offsets = offset_mapping[i]

                # sequence_ids tells which tokens are question(0)/context(1)/special(None)
                seq_ids = tokenized.sequence_ids(i)

                # For evaluation we keep offsets for context-only tokens
                if not is_train:
                    # mask non-context offsets as None (like HF examples)
                    clean_offsets = []
                    for o, sid in zip(offsets, seq_ids):
                        if sid == (1 if self.pad_on_right else 0):
                            clean_offsets.append(o)
                        else:
                            clean_offsets.append((0, 0))
                    features.append({
                        "id": ex_id,
                        "doc_id": doc_id,
                        "question": q,
                        "context": c,
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "offset_mapping": clean_offsets,
                    })
                    continue

                # Training: compute start/end positions
                if answer_text is None or answer_start is None or answer_end is None:
                    # unanswerable: convention => start=end=CLS
                    cls_index = input_ids.index(self.tokenizer.cls_token_id)
                    features.append({
                        "id": ex_id,
                        "doc_id": doc_id,
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "start_positions": cls_index,
                        "end_positions": cls_index,
                    })
                    continue

                # Identify the context token span in this feature
                context_token_indices = [idx for idx, sid in enumerate(seq_ids) if sid == (1 if self.pad_on_right else 0)]
                if not context_token_indices:
                    cls_index = input_ids.index(self.tokenizer.cls_token_id)
                    features.append({
                        "id": ex_id,
                        "doc_id": doc_id,
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "start_positions": cls_index,
                        "end_positions": cls_index,
                    })
                    continue

                token_start_index = context_token_indices[0]
                token_end_index = context_token_indices[-1]

                # Does the answer lie inside this window?
                # Offsets are relative to the context string (for context tokens).
                # Find first/last token whose offsets cover the answer.
                if offsets[token_start_index][0] > answer_start or offsets[token_end_index][1] < answer_end:
                    # answer not fully inside this chunk => label as no-answer for this chunk
                    cls_index = input_ids.index(self.tokenizer.cls_token_id)
                    features.append({
                        "id": ex_id,
                        "doc_id": doc_id,
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "start_positions": cls_index,
                        "end_positions": cls_index,
                    })
                    continue

                # Move token_start_index forward to answer start
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= answer_start:
                    token_start_index += 1
                start_pos = token_start_index - 1

                # Move token_end_index backward to answer end
                while offsets[token_end_index][1] >= answer_end and token_end_index >= 0:
                    token_end_index -= 1
                end_pos = token_end_index + 1

                features.append({
                    "id": ex_id,
                    "doc_id": doc_id,
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "start_positions": start_pos,
                    "end_positions": end_pos,
                })

        return Dataset.from_list(features)
