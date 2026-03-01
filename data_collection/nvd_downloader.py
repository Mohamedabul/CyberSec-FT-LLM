import requests
import json
import os
import time
from datetime import datetime, timedelta

BASE_URL    = "https://services.nvd.nist.gov/rest/json/cves/2.0"
SAVE_DIR    = "raw_data/nvd"
CHUNK_DAYS  = 110   # NVD max allowed = 120 days per request
os.makedirs(SAVE_DIR, exist_ok=True)

def get_date_chunks(year):
    """Split year into ≤110-day windows. NVD enforces 120-day max."""
    chunks, start = [], datetime(year, 1, 1)
    end_of_year   = datetime(year, 12, 31, 23, 59, 59)
    while start <= end_of_year:
        end = min(start + timedelta(days=CHUNK_DAYS - 1), end_of_year)
        # Confirmed working format: YYYY-MM-DDTHH:MM:SS.mmm (no Z)
        chunks.append((
            start.strftime("%Y-%m-%dT%H:%M:%S.000"),
            end.strftime("%Y-%m-%dT%H:%M:%S.999")
        ))
        start = end + timedelta(seconds=1)
    return chunks

YEARS = range(2020, 2026)          # 2020 → 2025
RESULTS_PER_PAGE = 2000
REQUEST_DELAY = 6                  # NVD rate limit: 5 req/30s without API key

for year in YEARS:
    save_path = os.path.join(SAVE_DIR, f"nvd_{year}.json")

    if os.path.exists(save_path):
        print(f"[{year}] Already exists — skipping.")
        continue

    print(f"\n[{year}] Downloading...")

    all_vulnerabilities = []

    for pub_start, pub_end in get_date_chunks(year):
        start_index   = 0
        total_results = 1

        while start_index < total_results:
            params = {
                "resultsPerPage": RESULTS_PER_PAGE,
                "startIndex":     start_index,
                "pubStartDate":   pub_start,
                "pubEndDate":     pub_end,
            }

            response = requests.get(BASE_URL, params=params, timeout=45)

            if response.status_code != 200:
                print(f"[{year}] Error: {response.status_code} | {pub_start[:10]}→{pub_end[:10]}")
                break

            data             = response.json()
            total_results    = data.get("totalResults", 0)
            vulnerabilities  = data.get("vulnerabilities", [])

            all_vulnerabilities.extend(vulnerabilities)
            print(f"[{year}] [{pub_start[:10]}→{pub_end[:10]}] {len(all_vulnerabilities)} / {total_results}")

            start_index += RESULTS_PER_PAGE
            time.sleep(REQUEST_DELAY)

    # Save year file
    with open(save_path, "w") as f:
        json.dump({"year": year, "count": len(all_vulnerabilities), "vulnerabilities": all_vulnerabilities}, f)

    print(f"[{year}] ✅ Saved → {save_path}")

print("\nAll years complete.")