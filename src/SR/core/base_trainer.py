from dataclasses import dataclass
from typing import Callable, Dict, Optional
import os, random
import numpy as np
import torch
from transformers import TrainingArguments, Trainer

@dataclass
class TrainConfig:
    output_dir: str
    lr: float = 2e-5
    epochs: int = 5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    grad_accum_steps: int = 1
    max_grad_norm: float = 1.0
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
    logging_steps: int = 50
    seed: int = 42
    metric_for_best_model: str = "f1"
    greater_is_better: bool = True

class BaseHFTrainer:
    def __init__(
        self,
        model,
        tokenizer,
        train_ds,
        eval_ds,
        compute_metrics: Callable,
        data_collator,
        config: TrainConfig,
        precision_flags: Dict[str, bool],
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.train_ds = train_ds
        self.eval_ds = eval_ds
        self.compute_metrics = compute_metrics
        self.data_collator = data_collator
        self.config = config
        self.precision_flags = precision_flags
        self.trainer: Optional[Trainer] = None

    def set_seed(self):
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)

    def build_args(self) -> TrainingArguments:
        os.makedirs(self.config.output_dir, exist_ok=True)
        return TrainingArguments(
            output_dir=self.config.output_dir,
            learning_rate=self.config.lr,
            num_train_epochs=self.config.epochs,
            weight_decay=self.config.weight_decay,
            warmup_ratio=self.config.warmup_ratio,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.grad_accum_steps,
            max_grad_norm=self.config.max_grad_norm,
            evaluation_strategy=self.config.eval_strategy,
            save_strategy=self.config.save_strategy,
            logging_steps=self.config.logging_steps,
            remove_unused_columns=False,
            load_best_model_at_end=True,
            metric_for_best_model=self.config.metric_for_best_model,
            greater_is_better=self.config.greater_is_better,
            report_to=[],
            save_total_limit=2,
            **self.precision_flags,
        )

    def fit(self):
        self.set_seed()
        args = self.build_args()
        self.trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=self.train_ds,
            eval_dataset=self.eval_ds,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
        )
        return self.trainer.train()

    def evaluate(self):
        if not self.trainer:
            raise RuntimeError("Trainer not built. Call fit() first.")
        return self.trainer.evaluate()

    def predict(self, test_ds):
        if not self.trainer:
            raise RuntimeError("Trainer not built. Call fit() first.")
        return self.trainer.predict(test_ds)
