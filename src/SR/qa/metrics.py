from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

import numpy as np


def _normalize_answer(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^a-z0-9\s%./=-]", " ", s)  # keep some symbols useful in stats
    s = " ".join(s.split())
    return s

def _f1(pred: str, truth: str) -> float:
    pred_toks = _normalize_answer(pred).split()
    truth_toks = _normalize_answer(truth).split()
    if len(pred_toks) == 0 and len(truth_toks) == 0:
        return 1.0
    if len(pred_toks) == 0 or len(truth_toks) == 0:
        return 0.0
    common = {}
    for t in pred_toks:
        common[t] = common.get(t, 0) + 1
    num_same = 0
    for t in truth_toks:
        if common.get(t, 0) > 0:
            num_same += 1
            common[t] -= 1
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_toks)
    recall = num_same / len(truth_toks)
    return 2 * precision * recall / (precision + recall + 1e-9)

def _exact_match(pred: str, truth: str) -> float:
    return 1.0 if _normalize_answer(pred) == _normalize_answer(truth) else 0.0


class QAMetrics:
    """
    Computes EM/F1 for extractive QA.
    Assumes the Trainer's eval_pred includes:
      - predictions = (start_logits, end_logits)
      - label_ids   = (start_positions, end_positions)
    Also expects we can reconstruct text spans from offset_mapping if provided.
    """
    def __init__(self, tokenizer, features_eval: List[Dict[str, Any]]):
        self.tokenizer = tokenizer
        self.features_eval = features_eval  # keep raw feature dicts with offset_mapping, context, input_ids

    def __call__(self, eval_pred) -> Dict[str, float]:
        (start_logits, end_logits), (start_positions, end_positions) = eval_pred

        ems: List[float] = []
        f1s: List[float] = []

        for i in range(len(self.features_eval)):
            feat = self.features_eval[i]
            input_ids = feat["input_ids"]
            offsets = feat.get("offset_mapping")
            context = feat.get("context", "")

            # ground truth
            gt_start = int(start_positions[i])
            gt_end = int(end_positions[i])

            # predicted span
            pred_start = int(np.argmax(start_logits[i]))
            pred_end = int(np.argmax(end_logits[i]))
            if pred_end < pred_start:
                pred_end = pred_start

            # convert to text
            pred_text = self._span_to_text(context, offsets, pred_start, pred_end, input_ids)
            gold_text = self._span_to_text(context, offsets, gt_start, gt_end, input_ids)

            ems.append(_exact_match(pred_text, gold_text))
            f1s.append(_f1(pred_text, gold_text))

        return {
            "em": float(np.mean(ems) if ems else 0.0),
            "f1": float(np.mean(f1s) if f1s else 0.0),
        }

    def _span_to_text(self, context: str, offsets, start_idx: int, end_idx: int, input_ids) -> str:
        # Handle CLS-as-no-answer cases: return ""
        cls_id = self.tokenizer.cls_token_id
        if start_idx < 0 or end_idx < 0 or start_idx >= len(input_ids) or end_idx >= len(input_ids):
            return ""
        if input_ids[start_idx] == cls_id and input_ids[end_idx] == cls_id:
            return ""

        if offsets is None:
            return ""

        # offsets are (0,0) for non-context tokens
        s_char, _ = offsets[start_idx]
        _, e_char = offsets[end_idx]
        if s_char == 0 and e_char == 0:
            return ""
        if e_char < s_char:
            return ""
        return context[s_char:e_char]
