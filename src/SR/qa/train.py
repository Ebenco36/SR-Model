import argparse, os

from transformers import AutoTokenizer, AutoModelForQuestionAnswering, DefaultDataCollator

from src.SR.core.device import DeviceManager
from src.SR.core.base_trainer import TrainConfig, BaseHFTrainer
from src.SR.core.io import read_jsonl, write_jsonl, split_docs
from src.SR.core.registry import QA_MODELS
from src.SR.qa.dataset import QASlidingWindowFeaturizer
from src.SR.qa.metrics import QAMetrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="data/qa_train.jsonl")
    ap.add_argument("--model", required=True, choices=QA_MODELS)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_len", type=int, default=384)
    ap.add_argument("--stride", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    examples = read_jsonl(args.inp)
    train_ex, val_ex, test_ex = split_docs(examples, seed=args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    write_jsonl(os.path.join(args.out_dir, "train.jsonl"), train_ex)
    write_jsonl(os.path.join(args.out_dir, "val.jsonl"), val_ex)
    write_jsonl(os.path.join(args.out_dir, "test.jsonl"), test_ex)

    dm = DeviceManager(prefer_mps=True)
    device = dm.get_device()
    precision_flags = dm.recommended_precision(device)

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForQuestionAnswering.from_pretrained(args.model)

    featurizer = QASlidingWindowFeaturizer(
        tokenizer=tokenizer,
        max_len=args.max_len,
        doc_stride=args.stride,
        pad_on_right=True,
    )

    train_ds = featurizer.featurize(train_ex, is_train=True)

    # For eval: keep offsets/context to reconstruct predictions for EM/F1
    val_features = featurizer.featurize(val_ex, is_train=False)
    # Convert val_features Dataset -> list of dicts so our metric can access raw fields
    val_features_list = [val_features[i] for i in range(len(val_features))]

    # But the Trainer needs start/end labels to compute loss during eval as well
    val_ds = featurizer.featurize(val_ex, is_train=True)

    collator = DefaultDataCollator()
    metrics = QAMetrics(tokenizer=tokenizer, features_eval=val_features_list)

    cfg = TrainConfig(
        output_dir=args.out_dir,
        lr=args.lr,
        epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        seed=args.seed,
        metric_for_best_model="f1",  # our QAMetrics returns "f1"
        greater_is_better=True,
    )

    trainer = BaseHFTrainer(
        model=model,
        tokenizer=tokenizer,
        train_ds=train_ds,
        eval_ds=val_ds,
        compute_metrics=metrics,
        data_collator=collator,
        config=cfg,
        precision_flags=precision_flags,
    )

    trainer.fit()
    trainer.evaluate()


if __name__ == "__main__":
    main()


# python -m src.qa.train \
#   --in data/qa_train.jsonl \
#   --model microsoft/deberta-v3-base \
#   --out_dir runs/qa_deberta \
#   --epochs 3 \
#   --batch 8 \
#   --max_len 384 \
#   --stride 128
