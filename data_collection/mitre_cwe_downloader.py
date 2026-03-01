"""
mitre_cwe_downloader.py
-----------------------
Downloads the latest MITRE CWE XML dataset.
Saves raw XML to: raw_data/mitre/cwec_latest.xml
"""

import time
import logging
import zipfile
import requests
from pathlib import Path
from io import BytesIO

# ── Config ────────────────────────────────────────────────────
MITRE_CWE_URL = "https://cwe.mitre.org/data/xml/cwec_latest.xml.zip"
OUTPUT_DIR    = Path(__file__).resolve().parents[1] / "raw_data" / "mitre"
OUTPUT_XML    = OUTPUT_DIR / "cwec_latest.xml"
RETRY_LIMIT   = 3
RETRY_DELAY   = 5

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Download & Extract ────────────────────────────────────────
def download_cwe_xml() -> Path | None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if OUTPUT_XML.exists():
        log.info(f"Already exists — skipping: {OUTPUT_XML.name}")
        return OUTPUT_XML

    for attempt in range(1, RETRY_LIMIT + 1):
        try:
            log.info(f"Attempt {attempt}/{RETRY_LIMIT} — Fetching CWE XML...")
            response = requests.get(MITRE_CWE_URL, timeout=120)
            response.raise_for_status()
            content = BytesIO(response.content)
            log.info(f"Downloaded ({len(response.content) / 1024:.1f} KB)")
            break
        except requests.RequestException as e:
            log.warning(f"Attempt {attempt} failed: {e}")
            if attempt < RETRY_LIMIT:
                time.sleep(RETRY_DELAY)
            else:
                log.error("All attempts failed.")
                return None

    try:
        with zipfile.ZipFile(content) as zf:
            xml_files = [f for f in zf.namelist() if f.endswith(".xml")]
            if not xml_files:
                log.error("No XML found in zip.")
                return None
            with zf.open(xml_files[0]) as src, open(OUTPUT_XML, "wb") as dst:
                dst.write(src.read())
        log.info(f"✅ Saved → {OUTPUT_XML}")
        return OUTPUT_XML
    except zipfile.BadZipFile as e:
        log.error(f"Zip extraction failed: {e}")
        return None

if __name__ == "__main__":
    download_cwe_xml()
