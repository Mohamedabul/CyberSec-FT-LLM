"""
evaluate.py
-----------
Evaluates the fine-tuned model on val/test sets.
Reports: loss, perplexity, and sample predictions.
"""

import yaml
import math
import torch
from pathlib import Path
from transformers import AutoTokenizer

from data_loader import load_jsonl
from model_loader import load_model_for_inference

BASE = Path(__file__).resolve().parents[1]
with open(BASE / "configs" / "train_config.yaml") as f:
    cfg = yaml.safe_load(f)

mc = cfg["model"]
sc = cfg["save"]
dc = cfg["data"]

EVAL_SAMPLES = 200


def compute_perplexity(model, tokenizer, texts: list, max_len: int = 512):
    model.eval()
    total_loss, count = 0.0, 0
    with torch.no_grad():
        for text in texts:
            enc = tokenizer(
                text, return_tensors="pt", max_length=max_len,
                truncation=True, padding=False
            ).to(model.device)
            labels = enc["input_ids"].clone()
            out = model(**enc, labels=labels)
            total_loss += out.loss.item()
            count += 1
    avg_loss = total_loss / max(count, 1)
    return avg_loss, math.exp(avg_loss)


def format_for_eval(sample: dict, tokenizer, include_output: bool = True) -> str:
    instruction = sample.get("instruction", "")
    inp = sample.get("input", "")
    user_content = f"{instruction}\n\n{inp}" if inp else instruction
    messages = [{"role": "user", "content": user_content}]
    if include_output:
        messages.append({"role": "assistant", "content": sample.get("output", "")})
    return tokenizer.apply_chat_template(
        messages, tokenize=False,
        add_generation_prompt=not include_output
    )


def generate_response(model, tokenizer, prompt: str, max_new: int = 256) -> str:
    enc = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **enc, max_new_tokens=max_new,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = out[0][enc["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def main():
    adapter_dir = str(BASE / sc["adapter_dir"])
    model = load_model_for_inference(mc["name"], adapter_dir, mc["trust_remote_code"])
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    tokenizer.pad_token = tokenizer.eos_token

    for split_name, split_path in [("val", dc["val_file"]), ("test", dc["test_file"])]:
        print(f"\n--- {split_name.upper()} SET ---")
        data = load_jsonl(str(BASE / split_path))[:EVAL_SAMPLES]

        texts = [format_for_eval(s, tokenizer, include_output=True) for s in data]
        avg_loss, ppl = compute_perplexity(model, tokenizer, texts)
        print(f"  Loss      : {avg_loss:.4f}")
        print(f"  Perplexity: {ppl:.2f}")

        print("\n  Sample predictions (3):")
        for i, s in enumerate(data[:3]):
            prompt = format_for_eval(s, tokenizer, include_output=False)
            pred = generate_response(model, tokenizer, prompt)
            ref  = s.get("output", "")[:200]
            uid  = s.get("cve_id", s.get("exploit_id", "N/A"))
            print(f"\n  [{i+1}] {uid}")
            print(f"  Ref : {ref}...")
            print(f"  Pred: {pred[:200]}...")

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
