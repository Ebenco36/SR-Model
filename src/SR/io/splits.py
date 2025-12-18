from dataclasses import dataclass
from typing import Dict, List
import random

@dataclass(frozen=True)
class SplitConfig:
    train: float = 0.70
    dev: float = 0.15
    test: float = 0.15
    seed: int = 42

class Splitter:
    def __init__(self, cfg: SplitConfig):
        self.cfg = cfg

    def split_doc_ids(self, doc_ids: List[str]) -> Dict[str, List[str]]:
        assert abs((self.cfg.train + self.cfg.dev + self.cfg.test) - 1.0) < 1e-6
        rng = random.Random(self.cfg.seed)
        ids = doc_ids[:]
        rng.shuffle(ids)

        n = len(ids)
        n_train = int(self.cfg.train * n)
        n_dev = int(self.cfg.dev * n)

        return {
            "train": ids[:n_train],
            "dev": ids[n_train:n_train + n_dev],
            "test": ids[n_train + n_dev:],
        }
