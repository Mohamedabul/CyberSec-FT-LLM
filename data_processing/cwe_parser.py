"""
cwe_parser.py
-------------
Parses MITRE CWE XML and extracts structured weakness records.
Output: processed_data/cwe_lookup.json  (CWE-ID → fields dict)
"""

import json
import xml.etree.ElementTree as ET
from pathlib import Path

XML_PATH = Path(__file__).resolve().parents[1] / "raw_data" / "mitre" / "cwec_latest.xml"
OUT_PATH = Path(__file__).resolve().parents[1] / "processed_data" / "cwe_lookup.json"

NS = {"cwe": "http://cwe.mitre.org/cwe-7"}

def get_text(element, tag: str) -> str:
    """Safely get text from a child element."""
    child = element.find(tag, NS)
    return " ".join(child.itertext()).strip() if child is not None else ""

def parse_consequences(weakness) -> list:
    """Extract common consequences (impact scope + technical impact)."""
    consequences = []
    for cons in weakness.findall(".//cwe:Common_Consequences/cwe:Consequence", NS):
        scope  = [s.text for s in cons.findall("cwe:Scope", NS) if s.text]
        impact = [i.text for i in cons.findall("cwe:Impact", NS) if i.text]
        if scope or impact:
            consequences.append({"scope": scope, "impact": impact})
    return consequences

def parse_cwe(weakness) -> dict:
    """Parse a single Weakness element into a structured record."""
    cwe_id   = "CWE-" + weakness.get("ID", "")
    name     = weakness.get("Name", "")
    abstraction = weakness.get("Abstraction", "")

    description      = get_text(weakness, "cwe:Description")
    extended_desc    = get_text(weakness, "cwe:Extended_Description")
    likelihood       = get_text(weakness, "cwe:Likelihood_Of_Exploit")

    return {
        "cwe_id":           cwe_id,
        "name":             name,
        "abstraction":      abstraction,      # Base / Variant / Class / Compound
        "description":      description,
        "extended_description": extended_desc,
        "likelihood_of_exploit": likelihood,
        "consequences":     parse_consequences(weakness),
    }

def parse_all():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Parsing {XML_PATH.name}...")

    tree = ET.parse(XML_PATH)
    root = tree.getroot()

    weaknesses = root.findall(".//cwe:Weakness", NS)
    print(f"Found {len(weaknesses)} weakness entries.")

    cwe_lookup = {}
    for weakness in weaknesses:
        record = parse_cwe(weakness)
        cwe_lookup[record["cwe_id"]] = record

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(cwe_lookup, f, indent=2)

    print(f"✅ Done — {len(cwe_lookup)} CWE records saved to {OUT_PATH}")

if __name__ == "__main__":
    parse_all()
