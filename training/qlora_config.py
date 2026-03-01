"""
qlora_config.py
---------------
Builds BitsAndBytesConfig and LoraConfig from train_config.yaml settings.
"""

import torch
from transformers import BitsAndBytesConfig
from peft import LoraConfig


def get_bnb_config(qc: dict) -> BitsAndBytesConfig:
    """4-bit quantization config for QLoRA."""
    return BitsAndBytesConfig(
        load_in_4bit=qc["load_in_4bit"],
        bnb_4bit_quant_type=qc["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=qc["bnb_4bit_use_double_quant"],
    )


def get_lora_config(lc: dict) -> LoraConfig:
    """LoRA adapter configuration."""
    return LoraConfig(
        r=lc["r"],
        lora_alpha=lc["lora_alpha"],
        lora_dropout=lc["lora_dropout"],
        bias=lc["bias"],
        task_type=lc["task_type"],
        target_modules=lc["target_modules"],
    )
