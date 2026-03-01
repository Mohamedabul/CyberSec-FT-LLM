"""
linker.py
---------
Links CVE records with CWE enrichment and matching Exploit-DB records.
Uses CVE ID as the central join key.

Inputs:
  - processed_data/cve_records.jsonl
  - processed_data/cwe_lookup.json
  - processed_data/exploit_records.jsonl

Output:
  - processed_data/linked_records.jsonl  (CVE + CWE + Exploit, fully joined)
  - processed_data/exploit_only.jsonl    (Exploits with no matching CVE in NVD)
"""

import json
from pathlib import Path
from collections import defaultdict

PROCESSED = Path(__file__).resolve().parents[1] / "processed_data"

CVE_PATH     = PROCESSED / "cve_records.jsonl"
CWE_PATH     = PROCESSED / "cwe_lookup.json"
EXPLOIT_PATH = PROCESSED / "exploit_records.jsonl"
LINKED_PATH  = PROCESSED / "linked_records.jsonl"
EXPLOIT_ONLY = PROCESSED / "exploit_only.jsonl"

def load_jsonl(path: Path) -> list:
    records, bad = [], 0
    with open(str(path), "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.replace("\x00", "").strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                bad += 1
    if bad:
        print(f"  [warn] Skipped {bad} malformed lines in {path.name}")
    return records

def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def link_all():
    print("Loading CVE records...")
    cve_records = load_jsonl(CVE_PATH)

    print("Loading CWE lookup...")
    cwe_lookup = load_json(CWE_PATH)

    print("Loading Exploit records...")
    exploit_records = load_jsonl(EXPLOIT_PATH)

    # Index exploits by each CVE ID they reference
    exploit_index = defaultdict(list)
    for exploit in exploit_records:
        for cve_id in exploit.get("cve_ids", []):
            exploit_index[cve_id.upper()].append(exploit)

    # Index CVEs by ID
    cve_index = {r["cve_id"].upper(): r for r in cve_records}

    linked, exploit_only = 0, 0

    with open(LINKED_PATH, "w", encoding="utf-8") as lf, \
         open(EXPLOIT_ONLY, "w", encoding="utf-8") as ef:

        # --- Build linked records (CVE + CWE + Exploit) ---
        for cve in cve_records:
            cve_id = cve["cve_id"].upper()

            # Enrich with CWE data
            cwe_enrichments = []
            for cwe_id in cve.get("cwe_ids", []):
                cwe_data = cwe_lookup.get(cwe_id)
                if cwe_data:
                    cwe_enrichments.append({
                        "cwe_id":      cwe_data["cwe_id"],
                        "name":        cwe_data["name"],
                        "abstraction": cwe_data["abstraction"],
                        "description": cwe_data["description"],
                        "likelihood":  cwe_data["likelihood_of_exploit"],
                        "consequences":cwe_data["consequences"],
                    })

            # Attach matching exploits
            matching_exploits = exploit_index.get(cve_id, [])
            exploits_clean = [
                {
                    "exploit_id":   e["exploit_id"],
                    "title":        e["title"],
                    "platform":     e["platform"],
                    "type":         e["type"],
                    "date":         e["date"],
                    "description":  e["description"],
                    "exploit_code": e["exploit_code"],
                }
                for e in matching_exploits
            ]

            record = {
                **cve,
                "cwe_enrichment":  cwe_enrichments,
                "exploits":        exploits_clean,
                "has_exploit":     len(exploits_clean) > 0,
            }
            lf.write(json.dumps(record) + "\n")
            linked += 1

        # --- Exploit-only records (no CVE match in NVD) ---
        for exploit in exploit_records:
            has_match = any(
                cid.upper() in cve_index
                for cid in exploit.get("cve_ids", [])
            )
            if not has_match:
                ef.write(json.dumps(exploit) + "\n")
                exploit_only += 1

    # Stats
    with_exploit = sum(1 for c in cve_records if c["cve_id"].upper() in exploit_index)
    print(f"\n✅ Linking complete:")
    print(f"   Total CVE records linked : {linked}")
    print(f"   CVEs with exploit match  : {with_exploit}")
    print(f"   Exploit-only records     : {exploit_only}")
    print(f"   Saved → {LINKED_PATH}")
    print(f"   Saved → {EXPLOIT_ONLY}")

if __name__ == "__main__":
    link_all()
