#!/usr/bin/env python3
"""
Compile a summary table of admin level coverage for each country, comparing
the survey data (parquet) against shapefile boundaries.

Output columns:
  country_code, coarser_admin_var, coarser_adm_level, coarser_adm_name,
  finer_admin_var, finer_adm_level, finer_adm_name, cluster_var,
  coarser_n_in_data, finer_n_in_data,
  coarser_n_in_shapefile, finer_n_in_shapefile,
  coarser_shapefile_boundaryYear, finer_shapefile_boundaryYear,
  cluster_var, lat_var, lon_var
"""

import os
import re
import json
import pandas as pd

COUNTRY_DATA_DIR  = "/data/eop/country_data"
ADMIN_LEVELS_CSV  = "/data/eop/compiled_country_data/geographic_indicator_admin_levels_manual.csv"
SHAPEFILES_DIR    = "/data/eop/geo/shapefiles"
OUTPUT_PATH       = "/data/eop/compiled_country_data/geographic_indicator_admin_levels_programmatic.csv"

# Countries whose finer admin count comes from a humdata table rather than a shapefile,
# because no shapefile exists at that level.
# Format: {country_code: (adm_level_str, full_csv_path, note)}
HUMDATA_FINER_OVERRIDES = {
    "CIV": ("ADM3", "/data/eop/geo/humdata_tables/CIV/civ_admin_boundaries_civ_admin3.csv", "finer count from OCHA humdata table; no ADM2 shapefile available"),
}

# Parquet files that are splits/summaries, not the full household records
_SKIP_PARQUET = {"summary.parquet", "test.parquet", "train.parquet"}


def _is_missing(val):
    return pd.isna(val) or str(val).strip().lower() in ("none", "nan", "")


def _adm_to_subdir(adm_level):
    """Convert 'ADM1' -> 'admin1', 'ADM2' -> 'admin2', etc."""
    return adm_level.lower().replace("adm", "admin")


def get_shapefile_info(cc, adm_level):
    """Return (admUnitCount, boundaryYear) for cc at adm_level, or (None, None)."""
    if _is_missing(adm_level):
        return None, None
    subdir = os.path.join(SHAPEFILES_DIR, cc, _adm_to_subdir(adm_level))
    if not os.path.isdir(subdir):
        return None, None
    meta_files = [f for f in os.listdir(subdir) if f.endswith("-metaData.json")]
    if not meta_files:
        return None, None
    with open(os.path.join(subdir, meta_files[0])) as f:
        meta = json.load(f)
    adm_count = meta.get("admUnitCount")
    return (
        int(adm_count) if adm_count else None,
        meta.get("boundaryYear") or None,
    )


def get_humdata_count(cc, adm_level):
    """Return row count from the humdata table for cc at adm_level, or None."""
    override = HUMDATA_FINER_OVERRIDES.get(cc)
    if override is None or override[0] != adm_level:
        return None
    csv_path = override[1]
    if not os.path.isfile(csv_path):
        return None
    return len(pd.read_csv(csv_path))


def count_distinct_in_parquet(cc, var_name):
    """Count distinct non-null values of var_name in the survey parquet for cc.
    """
    if _is_missing(var_name):
        return None
    cleaned_dir = os.path.join(COUNTRY_DATA_DIR, cc, "cleaned")
    if not os.path.isdir(cleaned_dir):
        return None

    full_path = os.path.join(cleaned_dir, "full.parquet")
    if os.path.isfile(full_path):
        try:
            col = pd.read_parquet(full_path, columns=[var_name])[var_name]
            return int(col.nunique(dropna=True))
        except Exception:
            pass

    return None


def main():
    admin_df = pd.read_csv(ADMIN_LEVELS_CSV)

    country_dirs = {
        d for d in os.listdir(COUNTRY_DATA_DIR)
        if re.match(r"^[A-Z]{3}$", d) and os.path.isdir(os.path.join(COUNTRY_DATA_DIR, d))
    }

    rows = []
    for _, admin_row in admin_df.iterrows():
        csv_cc  = admin_row["country_code"]
        base_cc = csv_cc.split("-")[0]

        if base_cc not in country_dirs:
            print(f"  [skip] {csv_cc}: no directory {base_cc}")
            continue

        coarser_var  = admin_row["coarser_admin_var"]
        coarser_lvl  = admin_row["coarser_adm_level"]
        coarser_name = admin_row["coarser_adm_name"]
        finer_var    = admin_row["finer_admin_var"]
        finer_lvl    = admin_row["finer_adm_level"]
        finer_name   = admin_row["finer_adm_name"]
        cluster_var  = admin_row["cluster_var"]
        lat_var      = admin_row["lat_var"]
        lon_var      = admin_row["lon_var"]

        coarser_n_data = count_distinct_in_parquet(base_cc, coarser_var)
        finer_n_data   = count_distinct_in_parquet(base_cc, finer_var)

        coarser_n_shp, coarser_yr = get_shapefile_info(base_cc, coarser_lvl)

        humdata_finer = get_humdata_count(base_cc, finer_lvl)
        if humdata_finer is not None:
            finer_n_shp, finer_yr = humdata_finer, None
            notes = HUMDATA_FINER_OVERRIDES[base_cc][2]
        else:
            finer_n_shp, finer_yr = get_shapefile_info(base_cc, finer_lvl)
            notes = ""

        rows.append({
            "country_code":                   csv_cc,
            "coarser_admin_var":              coarser_var,
            "coarser_adm_level":              coarser_lvl,
            "coarser_adm_name":               coarser_name,
            "finer_admin_var":                finer_var,
            "finer_adm_level":                finer_lvl,
            "finer_adm_name":                 finer_name,
            "cluster_var":                    cluster_var,
            "lat_var":                        lat_var,
            "lon_var":                        lon_var,
            "coarser_n_in_data":              coarser_n_data,
            "finer_n_in_data":                finer_n_data,
            "coarser_n_in_shapefile":         coarser_n_shp,
            "finer_n_in_shapefile":           finer_n_shp,
            "coarser_shapefile_boundaryYear": coarser_yr,
            "finer_shapefile_boundaryYear":   finer_yr,
            "notes":                          notes,
        })

    result_df = pd.DataFrame(rows)
    result_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(result_df)} rows to {OUTPUT_PATH}\n")
    # print(result_df.to_string(index=False))


if __name__ == "__main__":
    main()
