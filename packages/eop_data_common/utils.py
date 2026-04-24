import pandas as pd
import pyreadstat
from glob import glob

import os

def get_latest_aux_data():

    aux_files = glob('/data/eop/compiled_country_data/auxiliary_data/auxiliary_data_*.csv')
    latest_file = max(aux_files, key=lambda x: x.split('_')[-1].split('.')[0])
    return pd.read_csv(latest_file)

def get_2021_currency_conversion_factor(country_code):

    conversion_factors = pd.read_csv('/data/eop/compiled_country_data/currency_conversion.csv')
    row = conversion_factors[conversion_factors['country_code'] == country_code]
    assert len(row) == 1, f'instead of one row for country {country_code}, {len(row)} found.'
    return row['Conversion Factor'].values[0]

def get_2017_currency_conversion_factor(country_code):
    
    aux_files = glob('/data/eop/compiled_country_data/auxiliary_data/auxiliary_data_*.csv')
    latest_file = max(aux_files, key=lambda x: x.split('_')[-1].split('.')[0])
    aux_data = pd.read_csv(latest_file)

    row = aux_data[aux_data['country_code'] == country_code]
    assert len(row) == 1, f'instead of one row for country {country_code}, {len(row)} found.'
    return row['overall_currency_conversion_to_2017_ppp'].values[0]


def get_survey_year_wb_rate(country_code, currency_year):
    assert str(currency_year) in ['2017', '2021']

    aux_files = glob('/data/eop/compiled_country_data/auxiliary_data/auxiliary_data_*.csv')
    latest_file = max(aux_files, key=lambda x: x.split('_')[-1].split('.')[0])
    aux_data = pd.read_csv(latest_file)

    row = aux_data[aux_data['country_code'] == country_code]
    assert len(row) == 1, f'instead of one row for country {country_code}, {len(row)} found.'
    return row[f'wb_poverty_rate_povertyline_{currency_year}_survey_year'].values[0]

# Separate function to produce consumption using custom deflators from Elizabeth Foster.
def get_ehcvm_consumption(country_code, survey_year):

    conversion_factors = pd.read_csv('/data/eop/compiled_country_data/currency_conversion.csv')
    row = conversion_factors[conversion_factors['country_code'] == country_code]
    assert len(row) == 1
    currency_factor = row['Conversion Factor'].values[0]

    welfare_files = [
        f for f in os.listdir('/data/eop/other/ehcvm/factors_and_deflators') if (
            country_code.lower() in f.lower()
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