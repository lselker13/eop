import pandas as pd
import pyreadstat

import os

def get_2021_currency_conversion_factor(country_directory_name):

    conversion_factors = pd.read_csv('/data/eop/compiled_country_data/currency_conversion.csv')
    row = conversion_factors[conversion_factors['Country directory name'] == country_directory_name]
    assert len(row) == 1
    return row['Conversion Factor'].values[0]


def get_ehcvm_consumption(country_directory_name, survey_year):

    conversion_factors = pd.read_csv('/data/eop/compiled_country_data/currency_conversion.csv')
    row = conversion_factors[conversion_factors['Country directory name'] == country_directory_name]
    assert len(row) == 1
    currency_factor = row['Conversion Factor'].values[0]

    country_code = row['country_code'].values[0].lower()

    welfare_files = [
        f for f in os.listdir('/data/eop/other/ehcvm/factors_and_deflators') if (
            country_code in f.lower()
            and str(survey_year) in f
        )
    ]
    assert len(welfare_files) == 1
    welfare_file_path = f'/data/eop/other/ehcvm/factors_and_deflators/{welfare_files[0]}'
    try:
        welfare, _ = pyreadstat.read_dta(welfare_file_path)
    except UnicodeDecodeError:
        welfare, _ = pyreadstat.read_dta(welfare_file_path, encoding='latin1')

    temporal_deflator_col = [col for col in welfare.columns if 'def_temp_prix' in col]
    assert len(temporal_deflator_col) == 1
    temporal_deflator_col = temporal_deflator_col[0]

    welfare['consumption_per_capita_per_day'] = (
        welfare.dtot * currency_factor / (welfare.hhsize * 365 * welfare[temporal_deflator_col])
    )

    return welfare[['hhid', 'consumption_per_capita_per_day']]