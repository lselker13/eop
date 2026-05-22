#!/usr/bin/env python3
"""
Produce a table of country codes and accessed dates of raw survey data,
and write the results as a new column in survey_countries.csv.
Date is extracted from the name of the data directory under raw/.
"""

import re
import pandas as pd
from pathlib import Path

COUNTRY_DATA_DIR = Path("/data/eop/country_data")
SURVEY_CSV = Path("/data/eop/compiled_country_data/survey_countries.csv")

# Countries with multiple raw directories: hardcode which one to use.
# None means leave the accessed date blank.
HARDCODED_DIRS = {
    "BGD": "hies_2022_from_anik_2025_08_12",
    "CAF": None,
    "COD": "egi_odd_2020_accessed_2026_03_09",
    "IDN": None,
    "IND": "hces_2022_23_accessed_20250112",
    "PAK": "hies_2019_accessed_202510",
    "TGO": "ehcvm_2018_19_accessed_20250212",
    "ZAF": "ies_2010_11_accessed_20250516",
}


def extract_date(dirname: str) -> str:
    """Extract an accessed/acquired date from a raw data directory name."""
    # 'accessed_YYYY_MM_DD'
    m = re.search(r"accessed_(\d{4}_\d{2}_\d{2})", dirname)
    if m:
        parts = m.group(1).split("_")
        return f"{parts[0]}-{parts[1]}-{parts[2]}"

    # 'accessed_YYYYMMDD'
    m = re.search(r"accessed_(\d{8})", dirname)
    if m:
        d = m.group(1)
        return f"{d[:4]}-{d[4:6]}-{d[6:]}"

    # 'accessed_YYYYMM'
    m = re.search(r"accessed_(\d{6})", dirname)
    if m:
        d = m.group(1)
        return f"{d[:4]}-{d[4:]}"

    # Date at end of name: YYYY_MM_DD
    m = re.search(r"(\d{4})_(\d{2})_(\d{2})$", dirname)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"

    # Date at end of name: YYYYMMDD
    m = re.search(r"(\d{8})$", dirname)
    if m:
        d = m.group(1)
        return f"{d[:4]}-{d[4:6]}-{d[6:]}"

    # Date at end of name: YYYYMM
    m = re.search(r"(\d{6})$", dirname)
    if m:
        d = m.group(1)
        return f"{d[:4]}-{d[4:]}"

    return ""


def is_country_code(name: str) -> bool:
    return bool(re.match(r"^[A-Z]{3}$", name))


def main():
    rows = []

    for country_dir in sorted(COUNTRY_DATA_DIR.iterdir()):
        country = country_dir.name
        if not is_country_code(country):
            continue

        raw_dir = country_dir / "raw"

        if country in HARDCODED_DIRS:
            chosen_dirname = HARDCODED_DIRS[country]
            if chosen_dirname is None:
                rows.append((country, ""))
                continue
            chosen_path = raw_dir / chosen_dirname
            if not chosen_path.exists():
                raise FileNotFoundError(
                    f"Hardcoded directory for {country} not found: {chosen_path}"
                )
            rows.append((country, extract_date(chosen_dirname)))
            continue

        if not raw_dir.exists():
            rows.append((country, ""))
            continue

        subdirs = [d for d in raw_dir.iterdir() if d.is_dir()]
        if not subdirs:
            rows.append((country, ""))
            continue

        if len(subdirs) > 1:
            raise ValueError(
                f"Unexpected multiple directories for {country} (not in HARDCODED_DIRS): "
                + ", ".join(d.name for d in sorted(subdirs))
            )

        rows.append((country, extract_date(subdirs[0].name)))

    dates = dict(rows)

    df = pd.read_csv(SURVEY_CSV)
    df["raw_data_accessed"] = df["country_code"].map(dates)
    df.to_csv(SURVEY_CSV, index=False)

    print(f"{'Country':<10} Accessed Date")
    print("-" * 30)
    for country, date in rows:
        print(f"{country:<10} {date}")


if __name__ == "__main__":
    main()
