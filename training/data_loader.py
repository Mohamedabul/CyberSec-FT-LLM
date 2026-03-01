"""
data_loader.py
--------------
Loads and formats JSONL datasets into HuggingFace Dataset objects.
"""

import json
from pathlib import Path
from datasets import Dataset


def load_jsonl(path: str) -> list:
    records = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.replace("\x00", "").strip()
            if line:
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass
    return records


def format_prompt(sample: dict, tokenizer) -> str:
    """Format a sample into Phi-3.5 chat template format."""
    instruction = sample.get("instruction", "")
    inp = sample.get("input", "")
    output = sample.get("output", "")
    user_content = f"{instruction}\n\n{inp}" if inp else instruction

    messages = [
        {"role": "user",      "content": user_content},
        {"role": "assistant", "content": output},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )


def get_datasets(train_path: str, val_path: str, tokenizer) -> tuple:
    """Returns (train_dataset, val_dataset) as HuggingFace Datasets."""
    print("Loading datasets...")
    train_raw = load_jsonl(train_path)
    val_raw   = load_jsonl(val_path)
    print(f"  Train: {len(train_raw)} | Val: {len(val_raw)}")

    train_ds = Dataset.from_list(
        [{"text": format_prompt(s, tokenizer)} for s in train_raw]
    )
    val_ds = Dataset.from_list(
        [{"text": format_prompt(s, tokenizer)} for s in val_raw]
    )
    return train_ds, val_ds
