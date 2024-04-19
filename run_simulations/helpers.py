import copy

import csv
import os

import pandas as pd
import numpy as np

from pathlib import Path

PATH_TO_DATA = Path("/home/selker/eop/data/malawi/malawi_cleaned_2016.parquet")
PATH_TO_DATA_SUMMARY = Path("/home/selker/eop/data/malawi/malawi_summary_2016.parquet")
SAVE_PATH = Path("/home/selker/eop/eop/run_simulations/results/")
CONVERSION_FACTORS = {"malawi": 0.003361742723912196}

def load_malawi_data(covariates):

    summary = pd.read_parquet(PATH_TO_DATA_SUMMARY)

    covariate_columns = []
    for covariate in covariates:
        covariate_columns += list(summary.loc[covariate]['columns'])
    
    columns_to_select = covariate_columns + ['outcome', 'hh_wgt'] 
    df = pd.read_parquet(PATH_TO_DATA)[columns_to_select]

    return df[covariate_columns].to_numpy(), df.outcome.to_numpy(), df.hh_wgt.to_numpy(), covariate_columns

def split_data(X, y, p, r=None):
    rng = np.random.RandomState(123456)
    permutation = rng.permutation(X.shape[0])
    index_train = permutation[: int(p * X.shape[0])]
    index_test = permutation[int(p * X.shape[0]) :]
    X_train = X[index_train]
    X_test = X[index_test]
    y_train = y[index_train]
    y_test = y[index_test]

    if r is not None:
        r_train = r[index_train]
        r_test = r[index_test]
    else:
        r_train = np.ones(y_train.shape) / len(y_train)
        r_test = np.ones(y_test.shape) / len(y_test)

    return (X_train, y_train, r_train), (X_test, y_test, r_test)

def write_result(results_file, result, extra_run_labels=None):
    """Writes results to a csv file."""
    result = result.copy()
    result.update(extra_run_labels)

    with open(results_file, "a+", newline="") as csvfile:
        field_names = result.keys()
        dict_writer = csv.DictWriter(csvfile, fieldnames=field_names)
        if os.stat(results_file).st_size == 0:
            dict_writer.writeheader()
        dict_writer.writerow(result)

