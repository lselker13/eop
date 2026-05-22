#!/usr/bin/env python3
"""
Download shapefiles from geoBoundaries for each country in /data/eop/country_data/.
Downloads all available admin levels for each country into per-level subdirectories:
  /data/eop/geo/shapefiles/{ISO3}/admin1/
  /data/eop/geo/shapefiles/{ISO3}/admin2/
  ...
"""

import os
import re
import time
import zipfile
import requests
from datetime import date

COUNTRY_DATA_DIR = "/data/eop/country_data"
OUTPUT_DIR = "/data/eop/geo/shapefiles"
BASE_API = "https://www.geoboundaries.org/api/current/gbOpen"
DATE_STR = date.today().strftime("%Y-%m-%d")

ADMIN_LEVELS = ["ADM1", "ADM2", "ADM3", "ADM4", "ADM5"]


def get_country_codes():
    """Return ISO 3166-1 alpha-3 codes found in the country data directory."""
    return sorted(
        d for d in os.listdir(COUNTRY_DATA_DIR)
        if re.fullmatch(r"[A-Z]{3}", d)
    )


def fetch_level(iso3, level):
    """Return the staticDownloadLink for iso3/level, or None if unavailable."""
    url = f"{BASE_API}/{iso3}/{level}/"
    try:
        r = requests.get(url, timeout=30)
        if r.status_code != 200:
            return None
        data = r.json()
        return data.get("staticDownloadLink") or None
    except (requests.RequestException, ValueError):
        return None


def download(url, dest):
    r = requests.get(url, stream=True, timeout=300)
    r.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size=65536):
            f.write(chunk)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    countries = get_country_codes()
    print(f"Countries found ({len(countries)}): {', '.join(countries)}\n")

    successes, failures = [], []

    for iso3 in countries:
        print(f"[{iso3}]")
        country_ok = True

        for level in ADMIN_LEVELS:
            subdir = os.path.join(OUTPUT_DIR, iso3, level.lower().replace("adm", "admin"))

            # Skip if already downloaded (readme.txt is written on success)
            if os.path.exists(os.path.join(subdir, "readme.txt")):
                print(f"  {level} — already present, skipping")
                continue

            dl_url = fetch_level(iso3, level)
            time.sleep(0.25)

            if not dl_url:
                print(f"  {level} — not available")
                continue

            os.makedirs(subdir, exist_ok=True)
            zip_path = os.path.join(subdir, f"{iso3}_{level}_{DATE_STR}.zip")
            try:
                print(f"  {level} — downloading...", end=" ", flush=True)
                download(dl_url, zip_path)
                size_kb = os.path.getsize(zip_path) // 1024
                with zipfile.ZipFile(zip_path) as zf:
                    zf.extractall(subdir)
                os.remove(zip_path)
                with open(os.path.join(subdir, "readme.txt"), "w") as f:
                    f.write(
                        f"Country:     {iso3}\n"
                        f"Admin level: {level}\n"
                        f"Access date: {DATE_STR}\n"
                        f"Source:      geoBoundaries (https://www.geoboundaries.org)\n"
                        f"API URL:     {BASE_API}/{iso3}/{level}/\n"
                    )
                print(f"done ({size_kb} KB)")
            except Exception as e:
                print(f"ERROR: {e}")
                if os.path.exists(zip_path):
                    os.remove(zip_path)
                country_ok = False

            time.sleep(0.5)

        if country_ok:
            successes.append(iso3)
        else:
            failures.append(iso3)

    print(f"\n{'='*50}")
    print(f"Completed: {len(successes)}/{len(countries)} countries with no errors")
    if failures:
        print(f"Errors in: {', '.join(failures)}")


if __name__ == "__main__":
    main()
