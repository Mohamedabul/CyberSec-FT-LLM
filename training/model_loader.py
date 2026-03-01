"""
model_loader.py
---------------
Loads the base model with 4-bit quantization and applies LoRA adapters.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, prepare_model_for_kbit_training, PeftModel


def load_tokenizer(model_name: str, trust_remote_code: bool = True, max_length: int = 1024):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=trust_remote_code
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = max_length
    return tokenizer


def load_base_model(model_name: str, bnb_config, trust_remote_code: bool = True):
    """Load model in 4-bit quantized mode ready for QLoRA training."""
    print(f"Loading base model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=trust_remote_code,
        dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)
    return model


def apply_lora(model, lora_config):
    """Attach LoRA adapters to the model and print trainable param count."""
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def load_model_for_inference(model_name: str, adapter_dir: str, trust_remote_code: bool = True):
    """Load base model + LoRA adapter for evaluation/inference (no quantization)."""
    print(f"Loading model for inference from: {adapter_dir}")
    base = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=trust_remote_code,
        attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    return model


def merge_and_save(model_name: str, adapter_dir: str, merged_dir: str, trust_remote_code: bool = True):
    """Merge LoRA weights into base model and save the full model."""
    print(f"Merging LoRA into base model...")
    base = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=trust_remote_code,
    )
    merged = PeftModel.from_pretrained(base, adapter_dir)
    merged = merged.merge_and_unload()
    merged.save_pretrained(merged_dir, safe_serialization=True)
    print(f"Merged model saved to: {merged_dir}")
