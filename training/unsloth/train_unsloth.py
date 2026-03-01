"""
train_unsloth.py
----------------
QLoRA fine-tuning using Unsloth — 2-5x faster than HuggingFace+TRL.
Designed for Google Colab (T4/A100 GPU).

Setup:
  pip install unsloth trl transformers datasets pyyaml
"""

import json
import yaml
import glob
import os
from pathlib import Path
from datasets import Dataset
from trl import SFTTrainer, SFTConfig

from model_loader_unsloth import load_model_and_tokenizer

# ── Config ────────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parents[2]

def load_yaml(fname):
    with open(BASE / "configs" / fname) as f:
        return yaml.safe_load(f)

model_cfg    = load_yaml("model_config.yaml")
data_cfg     = load_yaml("data_config.yaml")
train_cfg    = load_yaml("training_config.yaml")

# Merge into single cfg dict expected by model_loader_unsloth
cfg = {
    "model":        {**model_cfg["base_model"], "max_seq_length": model_cfg["tokenizer"]["max_length"]},
    "quantization": model_cfg["quantization"],
    "lora":         model_cfg["lora"],
}

tc = train_cfg["common"]
cc = train_cfg["colab"]

# Paths — use Colab/Drive paths if they exist, else fall back to local
DATASET_BASE = Path(cc.get("dataset_path", str(BASE / "dataset")))
train_path   = str(DATASET_BASE / "train.jsonl")
val_path     = str(DATASET_BASE / "val.jsonl")
adapter_dir  = cc["output_dir"]
merged_dir   = adapter_dir.replace("/adapter", "/merged")

# ── Dataset ───────────────────────────────────────────────────────────────────
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
    """Format using tokenizer's built-in chat template."""
    instruction = sample.get("instruction", "")
    inp         = sample.get("input", "")
    output      = sample.get("output", "")
    user_content = f"{instruction}\n\n{inp}" if inp else instruction

    messages = [
        {"role": "user",      "content": user_content},
        {"role": "assistant", "content": output},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )


# ── Load model ────────────────────────────────────────────────────────────────
model, tokenizer = load_model_and_tokenizer(cfg)

# ── Build datasets ────────────────────────────────────────────────────────────
print("Loading datasets...")
train_raw = load_jsonl(train_path)
val_raw   = load_jsonl(val_path)
print(f"  Train: {len(train_raw)} | Val: {len(val_raw)}")

train_ds = Dataset.from_list([{"text": format_prompt(s, tokenizer)} for s in train_raw])
val_ds   = Dataset.from_list([{"text": format_prompt(s, tokenizer)} for s in val_raw])
del train_raw, val_raw

# ── SFTConfig ─────────────────────────────────────────────────────────────────
os.makedirs(adapter_dir, exist_ok=True)

args = SFTConfig(
    output_dir=adapter_dir,
    num_train_epochs=tc["num_train_epochs"],
    per_device_train_batch_size=cc["per_device_train_batch_size"],
    gradient_accumulation_steps=cc["gradient_accumulation_steps"],
    learning_rate=tc["learning_rate"],
    lr_scheduler_type=tc["lr_scheduler_type"],
    warmup_steps=tc["warmup_steps"],
    weight_decay=tc["weight_decay"],
    bf16=cc["bf16"],
    gradient_checkpointing=tc["gradient_checkpointing"],
    optim=cc["optim"],
    logging_steps=tc["logging_steps"],
    save_steps=tc["save_steps"],
    eval_steps=tc["eval_steps"],
    eval_strategy="steps",
    save_total_limit=tc["save_total_limit"],
    load_best_model_at_end=tc["load_best_model_at_end"],
    report_to=tc["report_to"],
    seed=tc["seed"],
    dataset_text_field="text",
)

# ── Trainer ───────────────────────────────────────────────────────────────────
trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    args=args,
)

# ── Resume from checkpoint if available ───────────────────────────────────────
checkpoints = sorted(glob.glob(os.path.join(adapter_dir, "checkpoint-*")))
resume_from = checkpoints[-1] if checkpoints else None
if resume_from:
    print(f"\nResuming from: {resume_from}")
else:
    print("\nStarting fresh training...")

print("Starting Unsloth QLoRA training...")
trainer.train(resume_from_checkpoint=resume_from)

# ── Save adapter ──────────────────────────────────────────────────────────────
print(f"\nSaving LoRA adapter to: {adapter_dir}")
trainer.model.save_pretrained(adapter_dir)
tokenizer.save_pretrained(adapter_dir)
print("Adapter saved.")

# ── Merge LoRA + save full model ──────────────────────────────────────────────
from unsloth import FastLanguageModel
print(f"\nMerging LoRA into base model...")

merged_model, merged_tokenizer = FastLanguageModel.from_pretrained(
    model_name=adapter_dir,
    max_seq_length=cfg["model"]["max_seq_length"],
    dtype=None,
    load_in_4bit=True,
)
os.makedirs(merged_dir, exist_ok=True)
merged_model.save_pretrained_merged(
    merged_dir,
    merged_tokenizer,
    save_method="merged_16bit",    # Full precision merged model
)
print(f"Merged model saved to: {merged_dir}")
print("\nTraining complete!")
