import json, random
from typing import Any, Dict, List, Tuple

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out

def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def split_docs(docs, seed=42, train=0.8, val=0.1):
    import random
    rng = random.Random(seed)
    docs = docs[:]
    rng.shuffle(docs)
    n = len(docs)

    n_train = int(train * n)
    n_val = int(val * n)

    # guarantee non-empty splits when possible
    if n >= 3 and n_val == 0:
        n_val = 1
    if n >= 3 and n_train == n:
        n_train = n - 2  # leave room for val+test
    if n - (n_train + n_val) == 0 and n >= 3:
        n_train -= 1

    return docs[:n_train], docs[n_train:n_train+n_val], docs[n_train+n_val:]

