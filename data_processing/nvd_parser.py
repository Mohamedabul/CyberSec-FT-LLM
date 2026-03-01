"""
nvd_parser.py
-------------
Parses NVD JSON files (2020–2025) and extracts structured CVE fields.
Output: processed_data/cve_records.jsonl
"""

import json
import os
from pathlib import Path

RAW_DIR  = Path(__file__).resolve().parents[1] / "raw_data" / "nvd"
OUT_PATH = Path(__file__).resolve().parents[1] / "processed_data" / "cve_records.jsonl"

def extract_cvss_v3(cve_item: dict) -> dict:
    """Extract CVSS v3 metrics from a CVE item."""
    try:
        metrics = cve_item.get("metrics", {})
        cvss_data = (
            metrics.get("cvssMetricV31", [{}])[0].get("cvssData", {}) or
            metrics.get("cvssMetricV30", [{}])[0].get("cvssData", {})
        )
        return {
            "cvss_v3_score":           cvss_data.get("baseScore"),
            "cvss_v3_severity":        cvss_data.get("baseSeverity"),
            "attack_vector":           cvss_data.get("attackVector"),
            "attack_complexity":       cvss_data.get("attackComplexity"),
            "privileges_required":     cvss_data.get("privilegesRequired"),
            "user_interaction":        cvss_data.get("userInteraction"),
            "scope":                   cvss_data.get("scope"),
            "confidentiality_impact":  cvss_data.get("confidentialityImpact"),
            "integrity_impact":        cvss_data.get("integrityImpact"),
            "availability_impact":     cvss_data.get("availabilityImpact"),
        }
    except Exception:
        return {}

def extract_cwe_ids(cve_item: dict) -> list:
    """Extract CWE IDs from weakness descriptions."""
    cwe_ids = []
    for weakness in cve_item.get("weaknesses", []):
        for desc in weakness.get("description", []):
            value = desc.get("value", "")
            if value.startswith("CWE-"):
                cwe_ids.append(value)
    return list(set(cwe_ids))

def parse_cve(entry: dict) -> dict | None:
    """Parse a single CVE entry into a flat structured record."""
    cve = entry.get("cve", {})
    cve_id = cve.get("id", "")

    # Skip rejected CVEs
    if cve.get("vulnStatus", "") == "Rejected":
        return None

    # Get English description
    descriptions = cve.get("descriptions", [])
    description = next((d["value"] for d in descriptions if d.get("lang") == "en"), "")
    if not description:
        return None

    record = {
        "cve_id":      cve_id,
        "description": description,
        "published":   cve.get("published", ""),
        "modified":    cve.get("lastModified", ""),
        "vuln_status": cve.get("vulnStatus", ""),
        "cwe_ids":     extract_cwe_ids(cve),
    }
    record.update(extract_cvss_v3(cve))
    return record

def parse_all():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    total, skipped = 0, 0

    with open(OUT_PATH, "w", encoding="utf-8") as out_f:
        for json_file in sorted(RAW_DIR.glob("nvd_*.json")):
            print(f"Parsing {json_file.name}...")
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            for entry in data.get("vulnerabilities", []):
                record = parse_cve(entry)
                if record:
                    out_f.write(json.dumps(record) + "\n")
                    total += 1
                else:
                    skipped += 1

    print(f"\n✅ Done — {total} CVE records saved to {OUT_PATH}")
    print(f"   Skipped: {skipped} (rejected or no description)")

if __name__ == "__main__":
    parse_all()
