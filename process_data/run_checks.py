"""
Run check_finished_data checks for every country.
Captures the key outputs needed for the summary table.
"""
import pandas as pd
from pathlib import Path
import numpy as np
import subprocess
import os
import sys
import io
import contextlib
from glob import glob

sys.path.insert(0, '/home/selker/eop/poverty/package/src')
from opt_targeted_transfers.dataset_utils import Dataset, split, standardize
from sklearn.linear_model import LinearRegression

data_path = Path('/data/eop/country_data')

country_stratifier_map = {
    'BGD': 'DivCode',
    'BEN': 'region',
    'BFA': 'region',
    'BDI': 'province',
    'CAF': 'region',
    'COL': 'domain',
    'CIV': 'region',
    'COD': 'Province',
    'ETH': 'region',
    'GHA': 'region',
    'GNB': 'region',
    'IND': 'state',
    'IDN': 'regency_city_code',
    'KEN': 'county',
    'LBR': 'stratum',
    'MDG': 'region',
    'MWI': 'ea_id',
    'MLI': 'region',
    'MEX': 'federalEntity',
    'NAM': 'region',
    'NER': 's00q01',
    'NGA': 'ea_id',
    'PAK': 'province',
    'RWA': 'district',
    'SEN': 'region',
    'ZAF': 'province',
    'SDN': 'region',
    'TZA': 'region',
    'TLS': 'strata',
    'TGO': 'cluster_id',
    'UGA': 'region',
    'YEM': 'stratum',
    'ZWE': 'district',
    'SLE': '_district',
}

# ---------------------------------------------------------------------------
# Helpers (from notebook)
# ---------------------------------------------------------------------------

def all_column_names_containing(df, substring):
    return [col for col in df.columns if substring in col]


def convert_to_onehot(df, summary):
    if 'type' in summary.columns:
        data_type_col = 'type'
    elif 'data_type' in summary.columns:
        data_type_col = 'data_type'
    if 'covariate' in summary.columns:
        covariate_col = 'covariate'
    elif 'variable_name' in summary.columns:
        covariate_col = 'variable_name'

    categorical_columns = summary[summary[data_type_col] == 'categorical'][covariate_col].tolist()
    categorical_columns = [col for col in categorical_columns if col in df.columns]

    counts = {col: df[col].nunique(dropna=True) for col in categorical_columns}
    top5 = pd.Series(counts).sort_values(ascending=False).head(5) if counts else pd.Series(dtype=int)

    one_hot = pd.get_dummies(df[categorical_columns]).astype(np.float32)
    df = df.drop(columns=categorical_columns)
    new_df = pd.concat([df, one_hot], axis=1)
    return new_df, top5


def get_data_for_geo_extrapolation(data, summary):
    geo_cols = summary[summary['geographic_indicator'] == True]['variable_name'].tolist()
    coarse_geo_cols = summary[summary['geographic_indicator_coarser'] == True]['variable_name'].tolist()
    remove_for_coarse = list(set(geo_cols) - set(coarse_geo_cols))
    return data.drop(columns=remove_for_coarse)


def load_and_run(train_path, test_path, summary_path, outcome='consumption_per_capita_per_day',
                 weight='headcount_adjusted_hh_wgt'):
    def _load(path):
        d = pd.read_parquet(path)
        for col in ['hhid', 'case_id', 'hh_id', 'hh_wgt']:
            if col in d.columns:
                d = d.drop(columns=[col])
        return d.reset_index(drop=True)

    summary = pd.read_parquet(summary_path)
    data1 = get_data_for_geo_extrapolation(_load(train_path), summary)
    data2 = get_data_for_geo_extrapolation(_load(test_path), summary)

    all_data = pd.concat([data1, data2], ignore_index=True)
    all_data, _ = convert_to_onehot(all_data, summary)

    train_data = get_data_for_geo_extrapolation(_load(train_path), summary)
    test_data = get_data_for_geo_extrapolation(_load(test_path), summary)

    covs = [c for c in train_data.columns if c not in [outcome, weight]]

    train_data, top5_train = convert_to_onehot(train_data, summary)
    test_data, _ = convert_to_onehot(test_data, summary)

    # Align columns
    for col in set(all_data.columns) - set(train_data.columns):
        train_data[col] = 0.0
    for col in set(all_data.columns) - set(test_data.columns):
        test_data[col] = 0.0

    train_dataset = Dataset(train_data.astype('float32'), outcome=outcome, covs=covs, weight=weight)
    train_dataset, _ = split(train_dataset)

    X, y, r = train_dataset.get_data()
    X, _, _ = standardize(X)
    y, _, _ = standardize(y)
    model = LinearRegression(fit_intercept=True)
    model.fit(X, y, sample_weight=r)
    r2 = model.score(X, y, sample_weight=r)

    return r2, top5_train


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

aux_files = glob('/data/eop/compiled_country_data/auxiliary_data/auxiliary_data_*.csv')
latest_aux = max(aux_files, key=lambda x: x.split('_')[-1].split('.')[0])
compiled_data = pd.read_csv(latest_aux)

results = []

countries = sorted(country_stratifier_map.keys())

for country_code in countries:
    country_data_path = data_path / country_code / 'cleaned'
    if not country_data_path.exists():
        print(f"[{country_code}] SKIP: no cleaned/ directory")
        continue

    print(f"\n{'='*60}")
    print(f"[{country_code}]")
    print(f"{'='*60}")

    row = {'country': country_code, 'errors': [], 'warnings': [], 'notes': []}

    try:
        # ----------------------------------------------------------------
        # Read data & summary
        # ----------------------------------------------------------------
        data = pd.read_parquet(country_data_path / 'full.parquet')
        summary = pd.read_parquet(country_data_path / 'summary.parquet')
        if 'variable_description' not in summary.columns:
            summary['variable_description'] = summary['variable_name']

        row['n_samples'] = len(data)

        # ----------------------------------------------------------------
        # Summary <-> data match
        # ----------------------------------------------------------------
        data_cols = set(data.columns)
        summary_vars = set(summary['variable_name'])
        missing_in_data = summary_vars - data_cols
        missing_in_summary = data_cols - summary_vars
        allowed_missing = {'hhid', 'consumption_per_capita_per_day', 'hh_wgt', 'headcount_adjusted_hh_wgt'}
        real_missing_data = missing_in_data - allowed_missing
        real_missing_summary = missing_in_summary - allowed_missing
        if real_missing_data:
            row['errors'].append(f"In summary not in data: {real_missing_data}")
        if real_missing_summary:
            row['errors'].append(f"In data not in summary: {real_missing_summary}")

        # ----------------------------------------------------------------
        # NaN check
        # ----------------------------------------------------------------
        nan_cols = data.columns[data.isna().any()].tolist()
        if nan_cols:
            row['errors'].append(f"NaN in: {nan_cols}")
        else:
            row['notes'].append("No NaNs")

        # ----------------------------------------------------------------
        # Missingness indicators
        # ----------------------------------------------------------------
        missingness_cols_missing = [c for c in data.columns if c.endswith('_missing')]
        missingness_cols_m = [c for c in data.columns if c.endswith('_m')]
        missingness_cols = missingness_cols_missing + missingness_cols_m
        with_missingness = (
            [c[:-len('_missing')] for c in missingness_cols_missing] +
            [c[:-len('_m')] for c in missingness_cols_m]
        )

        if missingness_cols_m:
            row['warnings'].append(f"{len(missingness_cols_m)} cols use '_m' suffix")

        non_binary_miss = []
        for c in missingness_cols:
            unique_vals = set(data[c].dropna().unique())
            if not unique_vals.issubset({0, 1, 0.0, 1.0, True, False}):
                bad = unique_vals - {0, 1, 0.0, 1.0, True, False}
                non_binary_miss.append(f"{c}: {bad}")
        if non_binary_miss:
            row['errors'].append(f"Non-binary missingness cols: {non_binary_miss}")

        relevant_summary = summary[summary.variable_name.isin(with_missingness)]
        cat_with_miss = relevant_summary[relevant_summary.data_type == 'categorical']
        if len(cat_with_miss) > 0:
            row['errors'].append(f"Categorical cols with missingness indicator: {cat_with_miss.variable_name.tolist()}")

        num_no_miss = summary[
            (summary.data_type == 'numeric')
            & (~summary.variable_name.isin(with_missingness))
            & (~summary.variable_name.str.endswith('_missing'))
            & (~summary.variable_name.str.endswith('_m'))
        ].variable_name.tolist()
        # Exclude standard non-missing numerics
        expected_no_miss = {'consumption_per_capita_per_day', 'hh_wgt', 'headcount_adjusted_hh_wgt', 'hh_size'}
        num_no_miss_flagged = [v for v in num_no_miss if v not in expected_no_miss]
        row['num_no_missingness'] = num_no_miss_flagged

        # ----------------------------------------------------------------
        # Consumption & poverty rate
        # ----------------------------------------------------------------
        assert 'consumption_per_capita_per_day' in data.columns
        assert 'headcount_adjusted_hh_wgt' in data.columns
        row['consumption_mean'] = round(data.consumption_per_capita_per_day.mean(), 3)
        row['consumption_std'] = round(data.consumption_per_capita_per_day.std(), 3)

        comp_row = compiled_data[compiled_data.country_code == country_code]
        if len(comp_row) > 0:
            wb_2017 = comp_row.wb_poverty_rate_povertyline_2017_survey_year.values[0]
            wb_2021 = comp_row.wb_poverty_rate_povertyline_2021_survey_year.values[0]
            conv = comp_row.overall_conversion_factor_ratio_from_2021_to_2017.values[0]

            adj = data.consumption_per_capita_per_day * conv
            count_poor_2017 = data[adj < 2.15].headcount_adjusted_hh_wgt.sum()
            total = data.headcount_adjusted_hh_wgt.sum()
            rate_2017 = count_poor_2017 / total
            disc_2017 = rate_2017 - wb_2017

            count_poor_2021 = data[data.consumption_per_capita_per_day < 3].headcount_adjusted_hh_wgt.sum()
            rate_2021 = count_poor_2021 / total
            disc_2021 = rate_2021 - wb_2021

            row['poverty_disc_2017'] = round(disc_2017, 6)
            row['poverty_disc_2021'] = round(disc_2021, 6)
        else:
            row['warnings'].append("Country not in compiled_data")

        # ----------------------------------------------------------------
        # Suspicious columns
        # ----------------------------------------------------------------
        forbidden_patterns = {
            'mobile phone': r'mobile|cell.?phone|handphone',
            'internet': r'internet|\bonline\b',
            'income/transfers': r'\bincome\b|remittance|private.transfer|gift.receiv',
            'fertilizer': r'fertiliz|fertilis',
            'crop details': r'\bcrop\b|cultivat|\bharvest\b|\byield\b',
            'ethnicity/religion': r'relig|ethn|tribe|caste|\bsect\b',
            'food insecurity': r'skip.?meal|food.?insecur|\bhunger\b|\bstarv',
        }
        forbidden_found = []
        for reason, pattern in forbidden_patterns.items():
            matches = summary[
                summary.variable_name.str.contains(pattern, case=False, na=False, regex=True) |
                summary.variable_description.str.contains(pattern, case=False, na=False, regex=True)
            ]
            if len(matches) > 0:
                cols = matches.variable_name.tolist()
                forbidden_found.append(f"{reason}: {cols}")
        row['forbidden'] = forbidden_found

        # id/code columns listed as numeric
        id_code_numeric = summary[
            summary.variable_name.str.contains('id|code', case=False, na=False) &
            (summary.data_type == 'numeric')
        ].variable_name.tolist()
        row['id_code_numeric'] = id_code_numeric

        unit_cols = summary[
            summary.variable_name.str.contains('unit', na=False) |
            summary.variable_description.str.contains('unit', na=False)
        ]
        unit_cols = unit_cols[~unit_cols.variable_name.str.contains('community', na=False)]
        row['unit_cols'] = unit_cols.variable_name.tolist()

        # ----------------------------------------------------------------
        # Geographic indicators
        # ----------------------------------------------------------------
        geo_cols = summary[summary.geographic_indicator == True][['variable_name', 'variable_description', 'data_type']].copy()
        row['geo_cols'] = geo_cols.variable_name.tolist()

        # Columns that look geographic but are NOT marked as geo indicators
        geo_keywords = r'region|province|district|county|department|commune|urban|rural|location|zone|ward|division|subcounty|subregion|strata|stratum|municipality|governorate|wilaya|woreda|kebele|upazila|union|tehsil|mandal|taluka|enumerat|cluster|community|environment|milieu|setting'
        possible_geo_not_marked = summary[
            (
                summary.variable_name.str.contains(geo_keywords, case=False, na=False, regex=True) |
                summary.variable_description.str.contains(geo_keywords, case=False, na=False, regex=True)
            ) &
            (summary.geographic_indicator != True) &
            (~summary.variable_name.str.endswith('_missing')) &
            (~summary.variable_name.str.endswith('_m'))
        ]
        row['possible_geo_not_marked'] = possible_geo_not_marked[['variable_name', 'variable_description']].values.tolist()

        # ----------------------------------------------------------------
        # Data type check
        # ----------------------------------------------------------------
        dtype_errors = []
        for _, r_ in summary[summary.data_type == 'numeric'].iterrows():
            col = r_.variable_name
            if col in data.columns and not pd.api.types.is_numeric_dtype(data[col]):
                dtype_errors.append(col)
        row['dtype_errors'] = dtype_errors

        # ----------------------------------------------------------------
        # Split.sh
        # ----------------------------------------------------------------
        split_result = subprocess.run(
            ['bash', str(country_data_path / 'split.sh')],
            capture_output=True, text=True, cwd=str(country_data_path)
        )
        if split_result.returncode != 0:
            row['errors'].append(f"split.sh failed: {split_result.stderr[:200]}")
            row['r2'] = None
            row['top5_cat'] = None
        else:
            # ----------------------------------------------------------------
            # R² and top-5 categorical
            # ----------------------------------------------------------------
            train_path = country_data_path / 'train.parquet'
            test_path = country_data_path / 'test.parquet'
            full_path = country_data_path / 'full.parquet'
            summary_path = country_data_path / 'summary.parquet'

            n_train = len(pd.read_parquet(train_path))
            n_test = len(pd.read_parquet(test_path))
            n_full = len(pd.read_parquet(full_path))
            if n_train + n_test != n_full:
                row['errors'].append(f"train+test ({n_train}+{n_test}) != full ({n_full})")

            r2, top5 = load_and_run(train_path, test_path, summary_path)
            row['r2'] = round(r2, 4)
            row['top5_cat'] = {k: int(v) for k, v in top5.items()}
            max_cat = int(top5.max()) if len(top5) > 0 else 0
            row['max_cat_distinct'] = max_cat
            if max_cat >= 30:
                row['warnings'].append(f"Top categorical column has {max_cat} distinct values (>=30)")
            if r2 > 0.7:
                row['warnings'].append(f"R² = {r2:.4f} is high (>0.7)")

    except Exception as e:
        import traceback
        row['errors'].append(f"EXCEPTION: {e}\n{traceback.format_exc()[-500:]}")

    results.append(row)
    print(f"  n={row.get('n_samples','?')}  R²={row.get('r2','?')}  errors={len(row['errors'])}  warnings={len(row['warnings'])}")

# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------
import json
out_path = '/home/selker/eop/eop/process_data/check_results.json'
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nResults saved to {out_path}")
