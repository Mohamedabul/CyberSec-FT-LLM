"""
model_loader_unsloth.py
-----------------------
Loads model using Unsloth's FastLanguageModel.
2x faster loading, 60% less VRAM than standard HuggingFace.
For use in Google Colab / Linux environments.
"""

import yaml
from pathlib import Path

# Unsloth must be installed: pip install unsloth
from unsloth import FastLanguageModel

# Qwen2.5 LoRA target modules (attention + MLP layers)
QWEN_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",   # Attention
    "gate_proj", "up_proj", "down_proj",        # MLP
]


def load_model_and_tokenizer(cfg: dict):
    """
    Load base model + tokenizer using Unsloth's FastLanguageModel.

    Args:
        cfg: merged config dict with keys: model, quantization, lora

    Returns:
        (model, tokenizer) tuple with LoRA adapters attached
    """
    mc = cfg["model"]
    qc = cfg["quantization"]
    lc = cfg["lora"]

    print(f"[Unsloth] Loading: {mc['name']}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=mc["name"],
        max_seq_length=mc.get("max_seq_length", 512),
        dtype=None,                          # Auto-detect (bfloat16 on Ampere+)
        load_in_4bit=qc["load_in_4bit"],
        trust_remote_code=mc.get("trust_remote_code", True),
    )

    # Attach LoRA adapters
    # NOTE: lora_dropout MUST be 0 for Unsloth fast QKV/O/MLP patching
    model = FastLanguageModel.get_peft_model(
        model,
        r=lc["r"],
        lora_alpha=lc["lora_alpha"],
        lora_dropout=lc["lora_dropout"],     # Must be 0.0 for Unsloth fast mode
        bias=lc["bias"],
        target_modules=QWEN_TARGET_MODULES,
        use_gradient_checkpointing="unsloth",
        random_state=42,
        use_rslora=False,
    )

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model.print_trainable_parameters()
    return model, tokenizer


def load_for_inference(adapter_dir: str, max_seq_length: int = 512):
    """Load fine-tuned model for inference (Unsloth 2x faster mode)."""
    print(f"[Unsloth] Loading adapter for inference: {adapter_dir}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter_dir,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer
