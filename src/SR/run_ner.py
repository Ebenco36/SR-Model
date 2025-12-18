import json
from pathlib import Path
from typing import Dict, List

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification

from src.SR.core.device import DeviceManager
from src.SR.core.base_trainer import BaseHFTrainer, TrainConfig
from src.SR.core.registry import NER_MODELS
from src.SR.io.splits import Splitter, SplitConfig
from src.SR.ner.labels import build_label2id
from src.SR.ner.ner_dataset import NERExample, NERDatasetBuilder
from src.SR.ner.ner_metrics import NERMetrics


def read_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main():
    gold_path = "data/ner_gold_v1.jsonl"
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    rows = read_jsonl(gold_path)
    doc_ids = [r["doc_id"] for r in rows]

    splits = Splitter(SplitConfig(seed=42)).split_doc_ids(doc_ids)
    by_id = {r["doc_id"]: r for r in rows}

    label2id = build_label2id()
    id2label = {v: k for k, v in label2id.items()}
    builder = NERDatasetBuilder(label2id=label2id, max_length=256)

    device_mgr = DeviceManager(prefer_mps=True)
    device = device_mgr.get_device()
    precision_flags = device_mgr.recommended_precision(device)

    results = []
    for model_name in NER_MODELS:
        print(f"\n=== NER model: {model_name} | device: {device.type} | precision: {precision_flags} ===")

        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            num_labels=len(label2id),
            label2id=label2id,
            id2label=id2label,
        )

        def encode_split(split_name: str) -> List[Dict]:
            encs = []
            for did in splits[split_name]:
                r = by_id[did]
                ex = NERExample(doc_id=r["doc_id"], text=r["text"], spans=r.get("spans", []))
                encs.append(builder.encode(tokenizer, ex))
            return encs

        train_enc = encode_split("train")
        dev_enc = encode_split("dev")
        test_enc = encode_split("test")

        train_ds = Dataset.from_list(train_enc)
        dev_ds = Dataset.from_list(dev_enc)
        test_ds = Dataset.from_list(test_enc)

        collator = DataCollatorForTokenClassification(tokenizer)
        metrics = NERMetrics(id2label)

        cfg = TrainConfig(
            output_dir=f"checkpoints/ner/{model_name.replace('/', '_')}",
            lr=2e-5,
            epochs=5,
            per_device_train_batch_size=8 if device.type != "cuda" else 16,
            per_device_eval_batch_size=8 if device.type != "cuda" else 16,
            grad_accum_steps=4 if device.type == "mps" else 1,  # maximize MPS safely
            logging_steps=50,
        )

        trainer = BaseHFTrainer(
            model=model,
            tokenizer=tokenizer,
            train_ds=train_ds,
            eval_ds=dev_ds,
            compute_metrics=metrics,
            data_collator=collator,
            config=cfg,
            precision_flags=precision_flags,
        )

        trainer.fit()
        dev_metrics = trainer.evaluate()
        test_pred = trainer.predict(test_ds)
        test_metrics = metrics((test_pred.predictions, test_pred.label_ids))

        row = {
            "model": model_name,
            "device": device.type,
            "dev_f1": dev_metrics.get("eval_f1"),
            "dev_precision": dev_metrics.get("eval_precision"),
            "dev_recall": dev_metrics.get("eval_recall"),
            "test_f1": test_metrics["f1"],
            "test_precision": test_metrics["precision"],
            "test_recall": test_metrics["recall"],
        }
        results.append(row)

        print("DEV:", dev_metrics)
        print("TEST:", test_metrics)

    # Save CSV
    csv_path = out_dir / "ner_metrics.csv"
    cols = list(results[0].keys()) if results else []
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for r in results:
            f.write(",".join("" if r[c] is None else str(r[c]) for c in cols) + "\n")

    print(f"\nSaved: {csv_path}")


if __name__ == "__main__":
    main()
