"""
type_b_builder.py
-----------------
Builds Type B instruction samples: Exploit-only technical explanation.
Input : processed_data/exploit_only.jsonl  (exploits without NVD CVE match)
        + processed_data/exploit_records.jsonl (all exploits, for coverage)
Output: dataset/type_b.jsonl
"""
import json
from pathlib import Path

EXPLOIT_ONLY = Path(__file__).resolve().parents[1] / "processed_data" / "exploit_only.jsonl"
OUT_PATH     = Path(__file__).resolve().parents[1] / "dataset" / "type_b.jsonl"

MAX_CODE_LEN  = 3000   # Truncate very long exploit codes
MAX_RECORDS   = float('inf')  # Cap Type B samples

INSTRUCTION = (
    "You are a reverse engineering and malware analysis expert. "
    "Analyze the following exploit code and provide a detailed technical explanation "
    "covering: what the exploit does, the vulnerability it targets, the attack technique used, "
    "potential impact on the victim system, and indicators of compromise."
)

def truncate(text: str, max_len: int = MAX_CODE_LEN) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + f"\n... [truncated — {len(text) - max_len} chars omitted]"

def format_input(exploit: dict) -> str:
    lines = [f"Exploit Title: {exploit.get('title', 'N/A')}"]
    if exploit.get("platform"):
        lines.append(f"Platform: {exploit['platform']}")
    if exploit.get("type"):
        lines.append(f"Type: {exploit['type']}")
    if exploit.get("cve_ids"):
        lines.append(f"Referenced CVEs: {', '.join(exploit['cve_ids'])}")
    if exploit.get("date"):
        lines.append(f"Published: {exploit['date']}")
    lines.append(f"\n--- Exploit Code ---\n{truncate(exploit.get('exploit_code', ''))}")
    return "\n".join(lines)

def format_output(exploit: dict) -> str:
    title    = exploit.get("title", "Unknown Exploit")
    platform = exploit.get("platform", "Unknown")
    etype    = exploit.get("type", "Unknown")
    desc     = exploit.get("description", "").strip()
    cves     = ", ".join(exploit.get("cve_ids", [])) or "No CVE reference"

    return (
        f"## Exploit Analysis: {title}\n\n"
        f"**Classification**: {etype.upper()} exploit targeting {platform}\n\n"
        f"**CVE References**: {cves}\n\n"
        f"**Technical Summary**: {desc if desc else 'This exploit targets a vulnerability in the specified platform.'}\n\n"
        f"**Attack Technique**: The exploit leverages low-level memory manipulation or "
        f"improper input validation to gain unauthorized access or cause unexpected behavior.\n\n"
        f"**Potential Impact**:\n"
        f"- {'Remote code execution or denial of service on target system.' if etype in ('remote', 'dos') else 'Local privilege escalation or unauthorized access.'}\n"
        f"- Compromise of system integrity and confidentiality.\n\n"
        f"**Indicators of Compromise (IoCs)**:\n"
        f"- Unusual process spawning from vulnerable application\n"
        f"- Unexpected network connections or shell processes\n"
        f"- Crash logs or core dumps from the targeted service\n\n"
        f"**Mitigation**: Patch the affected software immediately. "
        f"Implement input validation and memory safety checks. "
        f"Use exploit mitigations such as ASLR, DEP/NX, and stack canaries."
    )

def build():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    total, skipped = 0, 0

    with open(str(EXPLOIT_ONLY), "r", encoding="utf-8", errors="replace") as inf, \
         open(str(OUT_PATH), "w", encoding="utf-8") as outf:
        for line in inf:
            line = line.replace("\x00", "").strip()
            if not line:
                continue
            try:
                exploit = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            code = exploit.get("exploit_code", "")
            if len(code) < 50:
                skipped += 1
                continue

            sample = {
                "type":        "B",
                "exploit_id":  exploit.get("exploit_id"),
                "instruction": INSTRUCTION,
                "input":       format_input(exploit),
                "output":      format_output(exploit),
            }
            outf.write(json.dumps(sample) + "\n")
            total += 1

            if total >= MAX_RECORDS:
                break

    print(f"✅ Type B — {total} samples saved to {OUT_PATH}")
    print(f"   Skipped: {skipped}")

if __name__ == "__main__":
    build()
