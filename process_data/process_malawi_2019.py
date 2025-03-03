import pandas as pd
from pathlib import Path
import numpy as np
import json
import warnings
import math

from typing import Dict, Tuple
from collections import defaultdict
import sklearn.linear_model as sklearn_linear_model
import sklearn.metrics as sklearn_metrics
import sklearn.model_selection as sklearn_model_selection
import sklearn.preprocessing as sklearn_preprocessing
import sklearn.feature_selection as sklearn_feature_selection
import sklearn.ensemble as sklearn_ensemble
import sklearn.decomposition as sklearn_decomposition
from sklearn.impute import SimpleImputer

import dask.dataframe as ddf

import matplotlib.pyplot as plt
import pyreadstat
from pandas.api.types import is_numeric_dtype


malawi_directory = Path('/data/eop/malawi')
malawi_raw_data_directory = malawi_directory / 'raw'

def columns_equal(df, col1, col2):
    c1 = df[col1]
    c2 = df[col2]

    if pd.api.types.is_numeric_dtype(c1) and pd.api.types.is_numeric_dtype(c2):
        return np.isclose(c1, c2, rtol=1e-4, equal_nan=True).all()
    else:
        try:
            eq = (c1 == c2).all()
        except TypeError:
            # mismatched categories -> this comparison raises a type error
            eq = False
        return eq

# Source: https://docs.google.com/spreadsheets/d/11I0U413LgiVYuvgPhVL1M-5bfJabCFql75tQWG551U0/edit#gid=0
currency_conversion_factor = 0.003361735405

survey_directory = malawi_raw_data_directory / 'MWI_2019_IHS-V_v06_M_Stata'

roster_path = survey_directory / 'HH_MOD_B'

covariate_info_table_path = malawi_raw_data_directory / 'covariate_info_table.csv'

file_list =  [
        survey_directory / f for f in (
            'HH_MOD_F', # household level data
            'HH_MOD_H', # food security
            'HH_MOD_N1', # household enterprises
            'HH_MOD_S2', # household credit
            'HH_MOD_T', # subjective assessment of well-being
            'HH_MOD_X', # ag and fisheries filter
            'ag_mod_a', # ownership of land
            'ag_mod_e3', # coupon use - rainy season
            'hh_mod_a_filt', # household identification
            'ihs5_consumption_aggregate', # consumption
            'householdgeovariables_ihs5' # geo
        )
    ]

description_overrides = {
    'hh_f35_2': (
        'Of the total cost of cellphone service for the household, '
        'how much was spent on internet for all household members?'
    ),
    'hh_f35_3': (
        'Of the total cost of cellphone service for the household, '
        'how much was spent on airtime for all household members?'
    ),
    'hh_g09': (
        'Over the past one week (7 days), did any people that you did '
        'not list as household members eat any meals in your household?'
    ),
    'hh_f41_2': (
        'The last time your toilet facility was emptied, where were the '
        'contents emptied to?'
    ),
    'hh_f26a': (
        'When you last paid for electricity, what length of time '
        'did that payment cover?'
    ),
    'hh_m00': (
        'Did your household own or rent any farm implements, machinery '
        'and/or structures, such as hand hoe, panga knife, treadle pump, '
        'ox cart, tractor, plough, generator, chicken house, storage house, '
        'barn, etc... in the last 12 months?'
    ),
    'hh_h02a': (
        'In the past 7 days, how many days have you or someone in your '
        'household had to rely on less preferred or less expensive food?'
    ),
    'hh_f03b': (
        'Time unit of estimate of the rent they could receive renting the '
        'property'
    ),
    'hh_h06_oth': (
        'Specify what was the other cause of this situation (referring to '
        'a selection of food-insecurity situations)'
    ),
    'hh_f27' : (
        'Although you do not have electricity in your dwelling, does your '
        'village / neighborhood have access to electricity provided by ESCOM?'
    ),
    'hh_f41_3': (
        'Where is your toilet facility located?'
    ),
    'hh_t05': (
        'Imagine six steps, where on the bottom, the first step, stand the '
        'poorest people, and on the highest step, the sixth, stand the rich. '
        'On which step are most of you today?'
    ),
    'hh_o0a': (
        'Does the household head or spouse have any biological sons and/or '
        'daughters who are 15 years old and over and do not live in this household?'
    ),
    'ssa_aez09': (
        'Agro-ecological Zone of the household'
    ),
    'hh_s13a': (
        'Who turned you down when you tried to borrow? (follow-up to a question '
        'about asking for credit; I can’t read the entire question)'
    ),
    'hh_t14': (
        'During the last 12 months, was there a time when you or others in your '
        'household were unable to eat healthy and nutritious food because of a '
        'lack of money or other resources?'
    ),
    'hh_t01': (
        'Concerning your households food consumption over the past one month, '
        'which of the following is true? (less than adequate, adequate, more '
        'than adequate)'
    ),
    'hh_t08': (
        'Which of the following is true? Your current income . . . (followed '
        'by a list of judgments as to sufficiency of income)'
    ),
    'hh_t15': (
        'During the last 12 months, was there a time when you or others in your '
        'household ate only a few kinds of foods because of a lack of money or other '
        'resources?'
    ),
    'hh_x02': (
        'What was the most recent rainy season? (2017/18 or 2018/19)'
    )
    
}

def clean_survey(
    year, 
    extra_modules: Dict[str, Tuple[pd.DataFrame, Dict[str, str]]]=None,
    columns_to_drop=None,
    save_full_input_covariate_list=False
):

    # extra_modules is a dict from module name to
    #   a pair containing a dataframe and
    #     a dict from covariate name to description

    def merge_and_clean(dataframe, malawi, dataset_name):
        
        if malawi is None:
            malawi = dataframe
        
        else:
            malawi = malawi.merge(dataframe, on='case_id', how='outer', suffixes=('_left', '_right'))
    
            for c in malawi.columns:
                if c.endswith('_left'):
                    c_left = c
                    base = c_left[:-5]
                    c_right = f'{base}_right'
    
                    match = columns_equal(malawi, c_left, c_right)
                    
                    if match:
                        malawi.drop(columns=c_right, inplace=True)
                        malawi.rename(columns={c_left: base}, inplace=True)
                    # geographies are sometimes named and sometimes encoded as integers. If we've got one of each,  
                    # keep the string name: that way it won't accidentally be treated as numeric later.
                    elif (
                        (base in ['region', 'district'])
                        & (
                            pd.api.types.is_numeric_dtype(malawi[c_left]) 
                            + pd.api.types.is_numeric_dtype(malawi[c_right]) 
                            == 1
                          )
                    ):
                        if pd.api.types.is_numeric_dtype(malawi[c_left]):
                            malawi.drop(columns=c_left, inplace=True)
                            malawi.rename(columns={c_right: base}, inplace=True)
                        else:
                            malawi.drop(columns=c_right, inplace=True)
                            malawi.rename(columns={c_left: base}, inplace=True)
                    else:
                        print(f'error merging {dataset_name}, mismatch in {base}')
                        malawi.drop(columns=c_right, inplace=True)
                        malawi.rename(columns={c_left: base}, inplace=True)

        return malawi

    malawi = None
    covariate_labels_to_descriptions = dict()
    covariate_labels_to_modules = dict()

    # Read in survey files
    for file in file_list:

        with warnings.catch_warnings():
            warnings.simplefilter('ignore') # TODO: Investigate. Warning thrown from w/in pyreadstat.

            dataframe, metadata =  pyreadstat.read_dta(
                    f'{file}.dta', apply_value_formats=True
            )

        covariate_labels_to_descriptions.update(metadata.column_names_to_labels)

        for covariate_label in metadata.column_names_to_labels.keys():
            covariate_labels_to_modules[covariate_label] = file.name

        malawi = merge_and_clean(dataframe, malawi, file.name)

    # Add extra modules
    if extra_modules:
        for module_name, (dataframe, covariate_names_to_descriptions_for_module) in extra_modules.items():

            covariate_labels_to_descriptions.update(covariate_names_to_descriptions_for_module)
            malawi = merge_and_clean(dataframe, malawi, module_name)     
    
            for covariate_label in covariate_names_to_descriptions_for_module.keys():
                covariate_labels_to_modules[covariate_label] = module_name

    covariate_labels_to_descriptions.update(description_overrides)

    if save_full_input_covariate_list:
        covariates = malawi.columns.to_frame().reset_index(drop=True)
        covariates.columns = ['covariate']
        covariates['description'] = covariates['covariate'].apply(lambda c: covariate_labels_to_descriptions.get(c, c))
        covariates['module'] = covariates['covariate'].apply(lambda c: covariate_labels_to_modules.get(c,c))

        covariates.to_csv(malawi_directory / 'cleaned' / f'all_covariates.csv', index=False)

    if columns_to_drop:
        malawi.drop(columns=columns_to_drop, inplace=True)

    # Keep only durable and verifiable covariates
    covariate_info_table = pd.read_csv(covariate_info_table_path)
    covariate_info_table.durable = covariate_info_table.durable.fillna(0).astype(bool)
    covariate_info_table.known_categorical = covariate_info_table.known_categorical.fillna(0).astype(bool)

    durable_indicators = dict(zip(covariate_info_table.covariate, covariate_info_table.durable))
    columns_to_keep = [
        col for col in malawi.columns if (
            durable_indicators.get(col, False)
        ) or (
            col in ['rexpagg', 'case_id', 'HHID', 'hh_wgt']
        )
    ]
    
    malawi = malawi[columns_to_keep]
    
    # drop entirely nan columns
    malawi.dropna(axis=1, how='all', inplace=True)

    # compute outcome
    ADULT_MIN_AGE = 18
    
    roster, _ =  pyreadstat.read_dta(
        f'{roster_path}.dta', apply_value_formats=True
    )

    roster.columns = [c.lower() for c in roster.columns]
    
    roster['adult'] = roster.hh_b05a >= ADULT_MIN_AGE
    hh_adult_counts = (
        roster[roster.adult].groupby('case_id')[['hhid']].count().rename(columns={'hhid': 'num_adults'})
    )
    hh_child_counts = (
        roster[~roster.adult].groupby('case_id')[['hhid']].count().rename(columns={'hhid': 'num_children'})
    )
    
    malawi = (
        malawi
        .merge(hh_adult_counts, how='left', on='case_id')
        .merge(hh_child_counts, how='left', on='case_id')
    )
    
    malawi[['num_adults', 'num_children']] = (
        malawi[['num_adults', 'num_children']].fillna(value=0)
    )

    assert (malawi.num_adults + malawi.num_children <= 0).sum() == 0
    
    malawi["consumption_per_capita_per_day"] = malawi["rexpagg"] * currency_conversion_factor 
    # Could weight children and adults differently here.
    malawi["consumption_per_capita_per_day"] /= (malawi.num_adults + malawi.num_children)
    malawi["consumption_per_capita_per_day"] /= 365
    malawi.drop(columns=['rexpagg'], inplace=True)

    # Drop rows that are missing critical fields which we don't want to impute.
    malawi.dropna(subset=['consumption_per_capita_per_day', 'HHID', 'hh_wgt'], inplace=True)

    covariate_labels_to_descriptions['consumption_per_capita_per_day'] = (
        'daily consumption per capita, in units of 2017 USD PPP'
    )
    
    # Sort numeric and categorical columns so we can impute missing values in the numeric
    #  columns.

    # coerce columns to numeric that can be coerced
    for c in malawi.columns:
        malawi[c] = pd.to_numeric(malawi[c], errors='ignore')
    
    # coerce known categorical columns to string
    known_categorical = covariate_info_table[covariate_info_table.known_categorical].covariate.values
    known_categorical = np.append(
        known_categorical, ['HHID', 'case_id']
    )
    for c in known_categorical:
        malawi[c] = malawi[c].astype(str)
    
    # Compile column summary (before imputing and one-hot encoding)
    missing_counts = malawi.isnull().sum() + (malawi == "").sum()  
    means = malawi.mean(skipna=True, numeric_only=True)
    medians = malawi.median(skipna=True, numeric_only=True)
    stds = malawi.std(skipna=True, numeric_only=True)
    
    summary = pd.concat((missing_counts, means, medians, stds), axis=1)
    summary.columns = ['missing_count', 'mean', 'median', 'std']
    summary.reset_index(names='covariate', inplace=True)

    summary['missing_fraction'] = summary.missing_count / len(malawi)

    summary['description'] = summary.covariate.apply(lambda c: covariate_labels_to_descriptions.get(c, c))
    summary['module'] = summary.covariate.map(covariate_labels_to_modules)
    
    summary.missing_fraction = summary.missing_fraction.round(2)
    summary['median'] = summary['median'].round(2)
    summary['mean'] = summary['mean'].round(2)
    summary['std'] = summary['std'].round(2)

    # Split into numeric and non-numeric columns
    malawi_numeric = malawi.select_dtypes(include=[np.number])
    malawi_non_numeric = malawi.select_dtypes(exclude=[np.number, np.datetime64])
    
    def get_covariate_type(cov):
        
        if cov in malawi_numeric.columns:
            return 'numeric'
        elif cov in malawi_non_numeric.columns:
            return 'categorical'
    
    def get_covariate_minimum(cov):

        if cov in malawi_numeric.columns:
            return malawi_numeric[cov].min()
        
        else:
            return np.nan

    summary['type'] = summary['covariate'].apply(get_covariate_type)
    summary['minimum'] = summary['covariate'].apply(get_covariate_minimum)

    covariate_to_columns_map = {
        covariate: [covariate] for covariate in summary.covariate
    }
    
    # impute missing values with the mean. If they have high missingness,
    # add a missingness-indicator column.
    MISSINGNESS_CUTOFF = 0.15
    covariates_over_cutoff = summary[summary.missing_fraction > MISSINGNESS_CUTOFF].covariate.values
    for covariate in malawi_numeric.columns:
        if covariate in covariates_over_cutoff:
            dummy_column = f'{covariate}_nan'
            malawi_numeric[dummy_column] = malawi_numeric[covariate].isna()
            covariate_to_columns_map[covariate].append(dummy_column)
    
    # This is different from what roshni does: She uses 0 to impute
    # if missingness is >15%. 
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    imputer.fit(malawi_numeric)
    
    columns = malawi_numeric.columns
    malawi_numeric = pd.DataFrame(imputer.transform(malawi_numeric))

    malawi_numeric.columns = columns

    malawi = malawi_numeric.join(malawi_non_numeric)

    # populate summary columns
    summary['columns'] = summary.covariate.map(covariate_to_columns_map)

    malawi.columns = [c.lower() for c in malawi.columns]
    summary.covariate = summary.covariate.str.lower()
    summary['columns'] = summary['columns'].apply(lambda x: [c.lower() for c in x])

    malawi.drop(columns=['case_id', 'hhid'], inplace=True)
    summary = summary[summary.covariate.isin(malawi.columns)]

    return malawi, summary


if __name__ == '__main__':
    extra_modules = dict()

    durable_goods, durable_goods_metadata = pyreadstat.read_dta(
        survey_directory / 'HH_MOD_L.dta',apply_value_formats=True
    )

    durable_goods_pivoted = durable_goods.pivot_table(
        index='case_id', 
        columns='hh_l02', 
        values='hh_l03', 
        aggfunc='sum', 
        fill_value=0,
        observed=True # to avoid a warning
    ).add_prefix('durable_asset_')
    durable_goods_pivoted.columns.name = None
    durable_goods_pivoted = durable_goods_pivoted.loc[:, durable_goods_pivoted.sum(axis=0) > 0]
    durable_goods_covariate_to_desciption = dict()

    for covariate in durable_goods_pivoted.columns:
        durable_goods_covariate_to_desciption[covariate] = f'number owned: {covariate}'


    ag_goods, ag_goods_metadata = pyreadstat.read_dta(
        survey_directory / 'HH_MOD_M.dta',apply_value_formats=True
    )

    ag_goods.hh_m0b = ag_goods.hh_m0b.astype(str)

    ag_goods.loc[ag_goods.hh_m0b == 'OTHER', 'hh_m0b'] = ag_goods[ag_goods.hh_m0b == 'OTHER']['hh_m0b_oth']

    ag_goods_pivoted = ag_goods.pivot_table(
        index='case_id', 
        columns='hh_m0b', 
        values='hh_m01', 
        aggfunc='sum', 
        fill_value=0,
        observed=True # to avoid a warning
    ).add_prefix('ag_asset_')
    ag_goods_pivoted.columns.name = None
    ag_goods_pivoted = ag_goods_pivoted.loc[:, ag_goods_pivoted.sum(axis=0) > 0]
    ag_goods_covariate_to_description = dict()

    for covariate in ag_goods_pivoted.columns:
        ag_goods_covariate_to_description[covariate] = f'number owned: {covariate}'

    durable_goods_pivoted.reset_index(inplace=True)
    ag_goods_pivoted.reset_index(inplace=True)

    extra_modules['HH_MOD_L_durable_goods'] = (durable_goods_pivoted, durable_goods_covariate_to_desciption)
    extra_modules['HH_MOD_M_ag_goods'] = (ag_goods_pivoted, ag_goods_covariate_to_description)

    malawi_2019, summary_2019 = clean_survey(
        2019, extra_modules, save_full_input_covariate_list=True
    )

    malawi_2019.to_parquet(
        malawi_directory / 'cleaned' / 'malawi_2019.parquet', index=False
    )

    summary_2019.to_parquet(
        malawi_directory / 'cleaned' / 'summary_2019.parquet', index=False
    )