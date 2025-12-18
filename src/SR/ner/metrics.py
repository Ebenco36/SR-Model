from typing import Dict, List
import numpy as np
from seqeval.metrics import f1_score, precision_score, recall_score

class NERMetrics:
    """
    Entity-level precision/recall/F1 using seqeval on BIO tags.
    Expects id2label mapping like {0:'O', 1:'B-COUNTRY', 2:'I-COUNTRY', ...}
    """
    def __init__(self, id2label: Dict[int, str]):
        self.id2label = id2label

    def __call__(self, eval_pred) -> Dict[str, float]:
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)

        true_labels: List[List[str]] = []
        true_preds: List[List[str]] = []

        for p_seq, l_seq in zip(preds, labels):
            t_l, t_p = [], []
            for p, l in zip(p_seq, l_seq):
                if l == -100:
                    continue
                t_l.append(self.id2label[int(l)])
                t_p.append(self.id2label[int(p)])
            if t_l:  # avoid empty sequences edge cases
                true_labels.append(t_l)
                true_preds.append(t_p)

        return {
            "precision": float(precision_score(true_labels, true_preds)),
            "recall": float(recall_score(true_labels, true_preds)),
            "f1": float(f1_score(true_labels, true_preds)),
        }
