"""
merge_split.py
--------------
Merges Type A + B + C datasets and splits into train/val/test sets.
Output:
  dataset/train.jsonl   (~80%)
  dataset/val.jsonl     (~10%)
  dataset/test.jsonl    (~10%)
  dataset/dataset_stats.json
"""
import json
import random
from pathlib import Path

DATASET_DIR = Path(__file__).resolve().parents[1] / "dataset"

TYPE_FILES = {
    "A": DATASET_DIR / "type_a.jsonl",
    "B": DATASET_DIR / "type_b.jsonl",
    "C": DATASET_DIR / "type_c.jsonl",
}

TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
# TEST gets the remainder

SEED = 42

def load_jsonl(path: Path) -> list:
    records = []
    with open(str(path), "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.replace("\x00", "").strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records

def write_jsonl(records: list, path: Path):
    with open(str(path), "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

def merge_and_split():
    all_records = []
    type_counts = {}

    for t, path in TYPE_FILES.items():
        if not path.exists():
            print(f"  [warn] {path.name} not found — skipping Type {t}")
            continue
        records = load_jsonl(path)
        type_counts[t] = len(records)
        all_records.extend(records)
        print(f"  Loaded Type {t}: {len(records)} records")

    print(f"\n  Total samples: {len(all_records)}")

    random.seed(SEED)
    random.shuffle(all_records)

    n = len(all_records)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)

    train = all_records[:n_train]
    val   = all_records[n_train:n_train + n_val]
    test  = all_records[n_train + n_val:]

    write_jsonl(train, DATASET_DIR / "train.jsonl")
    write_jsonl(val,   DATASET_DIR / "val.jsonl")
    write_jsonl(test,  DATASET_DIR / "test.jsonl")

    stats = {
        "total":       n,
        "type_counts": type_counts,
        "train":       len(train),
        "val":         len(val),
        "test":        len(test),
        "split":       f"{int(TRAIN_RATIO*100)}/{int(VAL_RATIO*100)}/{100 - int(TRAIN_RATIO*100) - int(VAL_RATIO*100)}",
        "seed":        SEED,
    }
    with open(str(DATASET_DIR / "dataset_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n✅ Split complete:")
    print(f"   Train : {len(train):,}")
    print(f"   Val   : {len(val):,}")
    print(f"   Test  : {len(test):,}")
    print(f"   Stats → {DATASET_DIR / 'dataset_stats.json'}")

if __name__ == "__main__":
    merge_and_split()
