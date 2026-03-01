"""
type_c_builder.py
-----------------
Builds Type C instruction samples: Combined CVE + Exploit attack chain reasoning.
Input : processed_data/linked_records.jsonl  (has_exploit=True rows only)
Output: dataset/type_c.jsonl
"""
import json
from pathlib import Path

IN_PATH  = Path(__file__).resolve().parents[1] / "processed_data" / "linked_records.jsonl"
OUT_PATH = Path(__file__).resolve().parents[1] / "dataset" / "type_c.jsonl"

MAX_CODE_LEN = 2000

INSTRUCTION = (
    "You are a cybersecurity expert specializing in vulnerability exploitation and attack chain analysis. "
    "Given the CVE details and associated exploit code below, provide a complete attack chain analysis "
    "covering: the root cause vulnerability, how the exploit leverages it, step-by-step attack flow, "
    "exploitation prerequisites, potential impact, and defensive countermeasures."
)

def truncate(text: str, max_len: int = MAX_CODE_LEN) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + f"\n... [truncated]"

def format_input(record: dict, exploit: dict) -> str:
    lines = [
        f"CVE ID: {record.get('cve_id', 'N/A')}",
        f"Description: {record.get('description', 'N/A')}",
    ]
    if record.get("cvss_v3_score"):
        lines.append(f"CVSS v3: {record['cvss_v3_score']} ({record.get('severity', 'N/A')})")
    if record.get("attack_vector"):
        lines.append(f"Attack Vector: {record['attack_vector']}")
    if record.get("cwe_ids"):
        lines.append(f"CWE: {', '.join(record['cwe_ids'])}")
    lines.append(f"\nExploit Title: {exploit.get('title', 'N/A')}")
    lines.append(f"Platform: {exploit.get('platform', 'N/A')} | Type: {exploit.get('type', 'N/A')}")
    code = exploit.get("exploit_code", "")
    if code:
        lines.append(f"\n--- Exploit Code ---\n{truncate(code)}")
    return "\n".join(lines)

def format_output(record: dict, exploit: dict) -> str:
    cve_id    = record.get("cve_id", "N/A")
    desc      = record.get("description", "")
    sev       = record.get("severity", "Unknown")
    av        = record.get("attack_vector", "Unknown")
    ac        = record.get("attack_complexity", "Unknown")
    pr        = record.get("privileges_required", "Unknown")
    cwes      = ", ".join(record.get("cwe_ids", [])) or "Unknown"
    e_title   = exploit.get("title", "Unknown Exploit")
    e_type    = exploit.get("type", "Unknown")
    platform  = exploit.get("platform", "Unknown")

    cwe_info = ""
    for cwe in record.get("cwe_enrichment", []):
        cwe_info = f"{cwe.get('cwe_id')} — {cwe.get('name', '')}: {cwe.get('description', '')[:300]}"
        break

    return (
        f"## Attack Chain Analysis: {cve_id} + {e_title}\n\n"
        f"**Root Cause Vulnerability**: {desc}\n\n"
        f"**Weakness Classification**: {cwes}\n"
        + (f"- {cwe_info}\n\n" if cwe_info else "\n")
        + f"**Severity**: {sev} | Attack Vector: {av} | Complexity: {ac} | Privileges: {pr}\n\n"
        f"**Exploit Overview**: '{e_title}' is a {e_type} exploit targeting {platform}. "
        f"It leverages the vulnerability described above to achieve unauthorized access or system compromise.\n\n"
        f"**Step-by-Step Attack Flow**:\n"
        f"1. Attacker identifies the vulnerable {platform} target.\n"
        f"2. {'No authentication required — attack is unauthenticated.' if pr == 'NONE' else 'Attacker obtains required access level.'}\n"
        f"3. Exploit payload is crafted and delivered via {av.lower()} access.\n"
        f"4. Vulnerability is triggered, leading to {e_type.upper()} condition.\n"
        f"5. Attacker achieves code execution or disrupts service availability.\n\n"
        f"**Prerequisites**: {av} access required. "
        f"{'No privileges needed.' if pr == 'NONE' else f'{pr} privileges required.'} "
        f"{'No user interaction.' if record.get('user_interaction') == 'NONE' else 'Requires user interaction.'}\n\n"
        f"**Impact**:\n"
        f"- Confidentiality: {record.get('confidentiality', 'Unknown')}\n"
        f"- Integrity: {record.get('integrity', 'Unknown')}\n"
        f"- Availability: {record.get('availability', 'Unknown')}\n\n"
        f"**Defensive Countermeasures**:\n"
        f"- Apply vendor security patches for {cve_id} immediately.\n"
        f"- {'Block inbound network traffic to vulnerable service.' if av == 'NETWORK' else 'Enforce least-privilege access controls.'}\n"
        f"- Deploy IDS/IPS signatures to detect exploit patterns.\n"
        f"- Audit logs for indicators: unusual process spawning, unexpected connections, crash dumps."
    )

def build():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    total, skipped = 0, 0

    with open(str(IN_PATH), "r", encoding="utf-8", errors="replace") as inf, \
         open(str(OUT_PATH), "w", encoding="utf-8") as outf:
        for line in inf:
            line = line.replace("\x00", "").strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            if not record.get("has_exploit"):
                continue
            if not record.get("description") or not record.get("cvss_v3_score"):
                skipped += 1
                continue

            for exploit in record.get("exploits", []):
                if len(exploit.get("exploit_code", "")) < 50:
                    continue

                sample = {
                    "type":        "C",
                    "cve_id":      record.get("cve_id"),
                    "exploit_id":  exploit.get("exploit_id"),
                    "instruction": INSTRUCTION,
                    "input":       format_input(record, exploit),
                    "output":      format_output(record, exploit),
                }
                outf.write(json.dumps(sample) + "\n")
                total += 1

    print(f"✅ Type C — {total} samples saved to {OUT_PATH}")
    print(f"   Skipped: {skipped}")

if __name__ == "__main__":
    build()
