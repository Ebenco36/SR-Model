import argparse, os
from transformers import AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification

from src.SR.core.device import DeviceManager
from src.SR.core.base_trainer import TrainConfig, BaseHFTrainer
from src.SR.core.io import read_jsonl, write_jsonl, split_docs
from src.SR.ner.dataset import build_bio_tagset, NerFeaturizer, NerSlidingWindowFeaturizer
from src.SR.ner.metrics import NERMetrics
from src.SR.core.registry import NER_MODELS

ATOMIC_LABELS = [
    "REVIEW_TYPE","STUDY_DESIGN","N_STUDIES","DATE_OF_LAST_LITERATURE_SEARCH","QUALITY_TOOL",
    "PATHOGEN","CANCER","CONDITION","HPV_TYPE","AGE_GROUP","GENDER","SPECIAL_POP","RISK_GROUP",
    "SAFETY","ACCEPTANCE","EFFICACY","IMMUNOGENICITY","COVERAGE","ECONOMIC","ADMINISTRATION",
    "ETHICAL","LOGISTICS","MODELLING","CLINICAL","LESION","COUNTRY","REGION","WHO_REGION",
    "INCOME_GROUP","VACCINE_TYPE","VACCINE_BRAND","DOSE","ROUTE","PROGRAM","COMPONENT",
    "SCREENING","COMBINATION","BARRIER","FACILITATOR","DATABASE","SEARCH_TERMS","INCLUSION",
    "ANALYSIS","PERIOD","FOLLOWUP","TIMING","SAMPLE_SIZE","PERCENT","COST","QALY","ICER",
    "EFFECT_MEASURE","EFFECT_VALUE","CI","PVALUE",
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--model", required=True, choices=NER_MODELS)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    docs = read_jsonl(args.inp)
    train_docs, val_docs, test_docs = split_docs(docs, seed=args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    write_jsonl(os.path.join(args.out_dir, "train.jsonl"), train_docs)
    write_jsonl(os.path.join(args.out_dir, "val.jsonl"), val_docs)
    write_jsonl(os.path.join(args.out_dir, "test.jsonl"), test_docs)

    dm = DeviceManager(prefer_mps=True)
    device = dm.get_device()
    precision_flags = dm.recommended_precision(device)

    tags, tag2id, id2tag = build_bio_tagset(ATOMIC_LABELS)

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    # featurizer = NerFeaturizer(tokenizer=tokenizer, tag2id=tag2id, max_len=args.max_len)
    featurizer = NerSlidingWindowFeaturizer(tokenizer=tokenizer, tag2id=tag2id, max_len=args.max_len, stride=128)

    train_ds = featurizer.featurize(train_docs)
    val_ds = featurizer.featurize(val_docs)

    model = AutoModelForTokenClassification.from_pretrained(
        args.model,
        num_labels=len(tags),
        id2label=id2tag,
        label2id=tag2id,
    )

    collator = DataCollatorForTokenClassification(tokenizer)
    metrics = NERMetrics(id2label=id2tag)

    cfg = TrainConfig(
        output_dir=args.out_dir,
        lr=args.lr,
        epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        seed=args.seed,
        metric_for_best_model="f1",
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


# python -m src.ner.train \
#   --in data/ner_gold_v1.jsonl \
#   --model allenai/scibert_scivocab_uncased \
#   --out_dir runs/ner_scibert \
#   --epochs 5 --batch 8 --max_len 512
