"""
type_a_builder.py
-----------------
Builds Type A instruction samples: CVE-only structured vulnerability analysis.
Input : processed_data/cve_records.jsonl
Output: dataset/type_a.jsonl
"""
import json
from pathlib import Path

IN_PATH  = Path(__file__).resolve().parents[1] / "processed_data" / "cve_records.jsonl"
OUT_PATH = Path(__file__).resolve().parents[1] / "dataset" / "type_a.jsonl"

INSTRUCTION = (
    "You are a cybersecurity expert. Analyze the following CVE and provide a "
    "structured vulnerability report covering: vulnerability type, affected component, "
    "attack vector, severity, potential impact, and recommended mitigations."
)

def format_input(cve: dict) -> str:
    lines = [f"CVE ID: {cve.get('cve_id', 'N/A')}"]
    lines.append(f"Description: {cve.get('description', 'N/A')}")
    if cve.get("cvss_v3_score"):
        lines.append(f"CVSS v3 Score: {cve['cvss_v3_score']} ({cve.get('severity', 'N/A')})")
    if cve.get("attack_vector"):
        lines.append(f"Attack Vector: {cve['attack_vector']}")
    if cve.get("attack_complexity"):
        lines.append(f"Attack Complexity: {cve['attack_complexity']}")
    if cve.get("privileges_required"):
        lines.append(f"Privileges Required: {cve['privileges_required']}")
    if cve.get("cwe_ids"):
        lines.append(f"CWE IDs: {', '.join(cve['cwe_ids'])}")
    return "\n".join(lines)

def format_output(cve: dict) -> str:
    score   = cve.get("cvss_v3_score", "Unknown")
    sev     = cve.get("severity", "Unknown")
    av      = cve.get("attack_vector", "Unknown")
    ac      = cve.get("attack_complexity", "Unknown")
    pr      = cve.get("privileges_required", "Unknown")
    ui      = cve.get("user_interaction", "Unknown")
    conf    = cve.get("confidentiality", "Unknown")
    integ   = cve.get("integrity", "Unknown")
    avail   = cve.get("availability", "Unknown")
    cwes    = ", ".join(cve.get("cwe_ids", [])) or "Not specified"
    desc    = cve.get("description", "")

    return (
        f"## Vulnerability Analysis: {cve.get('cve_id', 'N/A')}\n\n"
        f"**Severity**: {sev} (CVSS v3: {score})\n\n"
        f"**Vulnerability Summary**: {desc}\n\n"
        f"**Attack Characteristics**:\n"
        f"- Attack Vector: {av}\n"
        f"- Attack Complexity: {ac}\n"
        f"- Privileges Required: {pr}\n"
        f"- User Interaction Required: {ui}\n\n"
        f"**Impact Assessment**:\n"
        f"- Confidentiality Impact: {conf}\n"
        f"- Integrity Impact: {integ}\n"
        f"- Availability Impact: {avail}\n\n"
        f"**Weakness Classification**: {cwes}\n\n"
        f"**Recommended Mitigations**: Apply vendor patches immediately. "
        f"{'Restrict network access and require authentication.' if av == 'NETWORK' else 'Limit local user privileges.'} "
        f"Monitor for exploitation indicators."
    )

def build():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    total, skipped = 0, 0

    with open(str(IN_PATH), "r", encoding="utf-8") as inf, \
         open(str(OUT_PATH), "w", encoding="utf-8") as outf:
        for line in inf:
            line = line.strip()
            if not line:
                continue
            try:
                cve = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            # Require at minimum: description + CVSS score
            if not cve.get("description") or not cve.get("cvss_v3_score"):
                skipped += 1
                continue

            sample = {
                "type":        "A",
                "cve_id":      cve.get("cve_id"),
                "instruction": INSTRUCTION,
                "input":       format_input(cve),
                "output":      format_output(cve),
            }
            outf.write(json.dumps(sample) + "\n")
            total += 1

    print(f"✅ Type A — {total} samples saved to {OUT_PATH}")
    print(f"   Skipped: {skipped}")

if __name__ == "__main__":
    build()
