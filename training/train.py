"""
train.py
--------
Main entry point for QLoRA fine-tuning.
Imports from: data_loader, model_loader, qlora_config
"""

import yaml
from pathlib import Path
from trl import SFTTrainer, SFTConfig

from data_loader import get_datasets
from model_loader import load_tokenizer, load_base_model, apply_lora, merge_and_save
from qlora_config import get_bnb_config, get_lora_config

# ── Config ────────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parents[1]
with open(BASE / "configs" / "train_config.yaml") as f:
    cfg = yaml.safe_load(f)

mc = cfg["model"]
qc = cfg["quantization"]
lc = cfg["lora"]
tc = cfg["training"]
dc = cfg["data"]
sc = cfg["save"]

adapter_dir = str(BASE / sc["adapter_dir"])
merged_dir  = str(BASE / sc["merged_dir"])

# ── Load tokenizer ────────────────────────────────────────────────────────────
tokenizer = load_tokenizer(mc["name"], mc["trust_remote_code"])

# ── Load datasets ─────────────────────────────────────────────────────────────
train_ds, val_ds = get_datasets(
    train_path=str(BASE / dc["train_file"]),
    val_path=str(BASE / dc["val_file"]),
    tokenizer=tokenizer,
)

# ── Load model + apply QLoRA ──────────────────────────────────────────────────
bnb_config  = get_bnb_config(qc)
lora_config = get_lora_config(lc)
model = load_base_model(mc["name"], bnb_config, mc["trust_remote_code"])
model = apply_lora(model, lora_config)

# ── SFTConfig (Training args + SFT-specific settings) ───────────────────────
args = SFTConfig(
    output_dir=adapter_dir,
    num_train_epochs=tc["num_train_epochs"],
    per_device_train_batch_size=tc["per_device_train_batch_size"],
    gradient_accumulation_steps=tc["gradient_accumulation_steps"],
    learning_rate=tc["learning_rate"],
    lr_scheduler_type=tc["lr_scheduler_type"],
    warmup_steps=100,
    weight_decay=tc["weight_decay"],
    bf16=tc["bf16"],
    gradient_checkpointing=tc["gradient_checkpointing"],
    optim=tc["optim"],
    logging_steps=tc["logging_steps"],
    save_steps=tc["save_steps"],
    eval_steps=tc["eval_steps"],
    eval_strategy="steps",
    save_total_limit=tc["save_total_limit"],
    load_best_model_at_end=tc["load_best_model_at_end"],
    report_to=tc["report_to"],
    # SFT-specific
    dataset_text_field="text",
)

# ── Train ─────────────────────────────────────────────────────────────────────
trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    args=args,
)

print("\nStarting QLoRA training...")

# ── Auto-resume from latest checkpoint if one exists ──────────────────────────
import glob, os
checkpoints = sorted(glob.glob(os.path.join(adapter_dir, "checkpoint-*")))
resume_from = checkpoints[-1] if checkpoints else None
if resume_from:
    print(f"  Resuming from checkpoint: {resume_from}")
else:
    print("  No checkpoint found — starting fresh.")

trainer.train(resume_from_checkpoint=resume_from)


# ── Save adapter ──────────────────────────────────────────────────────────────
print(f"\nSaving LoRA adapter to: {adapter_dir}")
trainer.model.save_pretrained(adapter_dir)
tokenizer.save_pretrained(adapter_dir)
print("Adapter saved.")

# ── Merge and save full model ─────────────────────────────────────────────────
merge_and_save(mc["name"], adapter_dir, merged_dir, mc["trust_remote_code"])
tokenizer.save_pretrained(merged_dir)
print("\nTraining complete.")
