from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from datasets import Dataset


def build_bio_tagset(labels: List[str]) -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
    tags = ["O"]
    for lab in labels:
        tags += [f"B-{lab}", f"I-{lab}"]
    tag2id = {t: i for i, t in enumerate(tags)}
    id2tag = {i: t for t, i in tag2id.items()}
    return tags, tag2id, id2tag

@dataclass(frozen=True)
class NerSlidingWindowFeaturizer:
    tokenizer: Any
    tag2id: Dict[str, int]
    max_len: int = 512
    stride: int = 128

    LABEL_REMAP: Dict[str, str] = None  # type: ignore

    def __post_init__(self):
        if self.LABEL_REMAP is None:
            object.__setattr__(self, "LABEL_REMAP", {
                "SEARCH_DATE": "DATE_OF_LAST_LITERATURE_SEARCH",
            })

    def featurize(self, docs: List[Dict[str, Any]]) -> Dataset:
        rows = []
        for d in docs:
            text = d.get("text", "")
            if not isinstance(text, str) or not text.strip():
                continue
            spans = d.get("spans", [])

            enc = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_len,
                stride=self.stride,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding="max_length",
            )

            for i in range(len(enc["input_ids"])):
                offsets: List[Tuple[int, int]] = enc["offset_mapping"][i]
                labels = self._charspans_to_bio(offsets, spans, text_len=len(text))

                masked = []
                for (s, e), lab_id in zip(offsets, labels):
                    masked.append(-100 if s == e else int(lab_id))

                row = {
                    "input_ids": list(map(int, enc["input_ids"][i])),
                    "attention_mask": list(map(int, enc["attention_mask"][i])),
                    "labels": masked,
                }
                if "token_type_ids" in enc:
                    row["token_type_ids"] = list(map(int, enc["token_type_ids"][i]))

                rows.append(row)

        if not rows:
            raise RuntimeError("No training windows produced. Check inputs/max_len.")
        return Dataset.from_list(rows)

    def _charspans_to_bio(self, offsets: List[Tuple[int, int]], spans: List[Dict[str, Any]], text_len: int) -> List[int]:
        norm: List[Tuple[int, int, str]] = []
        for s in spans:
            try:
                s0 = int(s["start"])
                e0 = int(s["end"])
                lab = self.LABEL_REMAP.get(str(s["label"]), str(s["label"]))
            except Exception:
                continue
            if s0 < 0 or e0 > text_len or s0 >= e0:
                continue
            if f"B-{lab}" not in self.tag2id:
                continue
            norm.append((s0, e0, lab))

        # longest first
        norm.sort(key=lambda x: (-(x[1]-x[0]), x[0], x[1]))

        chosen: List[Tuple[int, int, str]] = []
        for s0, e0, lab in norm:
            ok = True
            for s1, e1, _ in chosen:
                if not (e0 <= s1 or s0 >= e1):
                    ok = False
                    break
            if ok:
                chosen.append((s0, e0, lab))

        out = [self.tag2id["O"]] * len(offsets)

        for s0, e0, lab in chosen:
            b_id = self.tag2id[f"B-{lab}"]
            i_id = self.tag2id[f"I-{lab}"]
            started = False
            for idx, (ts, te) in enumerate(offsets):
                if ts == te:
                    continue
                if te <= s0 or ts >= e0:
                    continue
                out[idx] = b_id if not started else i_id
                started = True

        return out
    
    
@dataclass(frozen=True)
class NerFeaturizer:
    tokenizer: Any
    tag2id: Dict[str, int]
    max_len: int = 512

    # put remap at class-level so it’s deterministic + easy to extend
    LABEL_REMAP: Dict[str, str] = None  # type: ignore

    def __post_init__(self):
        # dataclass frozen: use object.__setattr__
        if self.LABEL_REMAP is None:
            object.__setattr__(self, "LABEL_REMAP", {
                "SEARCH_DATE": "DATE_OF_LAST_LITERATURE_SEARCH",
            })

    def featurize(self, docs: List[Dict[str, Any]]) -> Dataset:
        rows = []
        for d in docs:
            text = d["text"]
            spans = d.get("spans", [])

            enc = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_len,
                stride=self.stride,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding="max_length",
            )


            offsets: List[Tuple[int, int]] = enc["offset_mapping"]
            labels = self._charspans_to_bio(offsets, spans, text_len=len(text))

            # mask special tokens
            masked = []
            for (s, e), lab_id in zip(offsets, labels):
                masked.append(-100 if s == e else int(lab_id))

            # IMPORTANT: remove offset_mapping, and ensure all fields are plain python lists
            enc.pop("offset_mapping", None)

            # Some tokenizers include token_type_ids, some don't. Keep only what exists.
            row = {
                "input_ids": list(map(int, enc["input_ids"])),
                "attention_mask": list(map(int, enc["attention_mask"])),
                "labels": masked,
            }
            if "token_type_ids" in enc:
                row["token_type_ids"] = list(map(int, enc["token_type_ids"]))

            # Hard checks: prevent bad rows from entering dataset
            L = len(row["input_ids"])
            if len(row["attention_mask"]) != L or len(row["labels"]) != L:
                continue  # skip malformed

            rows.append(row)

        if not rows:
            raise RuntimeError("No valid rows produced. Check your input JSONL and featurizer.")

        return Dataset.from_list(rows)

    def _charspans_to_bio(
        self,
        offsets: List[Tuple[int, int]],
        spans: List[Dict[str, Any]],
        text_len: int,
    ) -> List[int]:
        # Normalize + filter spans first
        norm: List[Tuple[int, int, str]] = []
        for s in spans:
            try:
                s0 = int(s["start"])
                e0 = int(s["end"])
                lab = str(s["label"])
            except Exception:
                continue

            # apply remap
            lab = self.LABEL_REMAP.get(lab, lab)

            # bounds
            if s0 < 0 or e0 > text_len or s0 >= e0:
                continue

            # drop unknown labels safely
            if f"B-{lab}" not in self.tag2id or f"I-{lab}" not in self.tag2id:
                continue

            norm.append((s0, e0, lab))

        # resolve overlaps: prefer longer spans first
        norm.sort(key=lambda x: (-(x[1] - x[0]), x[0], x[1]))

        chosen: List[Tuple[int, int, str]] = []
        for s0, e0, lab in norm:
            ok = True
            for s1, e1, _ in chosen:
                if not (e0 <= s1 or s0 >= e1):  # overlaps
                    ok = False
                    break
            if ok:
                chosen.append((s0, e0, lab))

        out = [self.tag2id["O"]] * len(offsets)

        for s0, e0, lab in chosen:
            b_id = self.tag2id[f"B-{lab}"]
            i_id = self.tag2id[f"I-{lab}"]

            started = False
            for idx, (ts, te) in enumerate(offsets):
                if ts == te:
                    continue  # special token
                if te <= s0 or ts >= e0:
                    continue  # no overlap
                out[idx] = b_id if not started else i_id
                started = True

        return out
