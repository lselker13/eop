import copy
import csv
from pathlib import Path
from functools import partial
from multiprocessing import Pool
import os

import argh
import numpy as np
import pandas as pd
from opt_targeted_transfers import (ConditionalTargetedTransfers,
                                    HybridTargetedTransfers,
                                    UnconditionalTargetedTransfers)

from helpers import write_result, load_malawi_data, split_data, SAVE_PATH


def malawi_runs(features, extra_run_labels):

    ds = [1, 3, 12, 20]
    poverty_rate_2016 = 0.6417
    tolerances = [0.1, 0.2]

    constraints = ["unconditional",]
    quantile_methods = ["qr", "density"]

    args = []

    for d in ds:
        features_d = features[:d]
        for constraint in constraints:
            if constraint == "conditional":
                for quantile_method in quantile_methods:

                    args.append(
                        {
                            'constraint': constraint, 
                            'features': features_d,
                            'method': quantile_method, 
                            'condtol': tolerances,
                        }
                    )

            elif constraint == "unconditional":

                args.append(
                        {
                            'constraint': constraint,  
                            'features': features_d,
                            'uncondtol': tolerances,
                        }
                )

            elif constraint == "hybrid":
                args.append(
                        {
                            'constraint': constraint,  
                            'features': features_d,
                            'uncondtol': tolerances,
                            'condtol': tolerances
                        }
                )
    for arg in args:
        arg.update({'extra_run_labels': extra_run_labels})
    pool = Pool()

    pool.map(
        unpack_and_run,
        args
    )
    # Probably unneeded because the above is synchronous.
    pool.close()
    pool.join()


def unpack_and_run(args_dictionary):
    return run(
        country='malawi', 
        save=SAVE_PATH,
        **args_dictionary
    )


def run(
    country="malawi",
    constraint="unconditional",
    method="qr",
    condtol=None,  # iterable
    uncondtol=None,  # iterable
    save="malawi_results.csv",
    features=None,
    extra_run_labels=None
):

    d = len(features)
    
    if extra_run_labels is None:
        extra_run_labels = dict()

    if features is None:
        # TODO: Redo this signature
        raise ValueError('Specify some features!')

    # A semi-hack to replace the incorrect d that was populated before.
    extra_run_labels.update({'d': d})
    
    X, y, r, _ = load_malawi_data(features)

    (X_train, y_train, r_train), (X_test, y_test, r_test) = split_data(
        X=X, y=y, r=r, p=0.6
    )

    base_name = f'{country}_d={d}_{constraint}'


    if constraint == "unconditional":
        tt = UnconditionalTargetedTransfers(c_bar=2.15)
        
        tt.fit(X_train, y_train, r_train)
        for tol in uncondtol:
            tt.set_unconditional_tolerance(tol)
            tt.run_opt(
                    X_test,
                    r_test,
                    path=save / f'{base_name}_uncondtol={tol}_opt.csv'
                )
            res = tt.evaluate(X_test, y_test, r_test)
            write_result(save / "{}.csv".format(country), res, extra_run_labels)
            tt.evaluate_equity(
                X_test,
                y_test,
                path=save / f'equity_{base_name}_uncondtol={tol}_opt.csv'
            )

    elif constraint == "conditional":
        tt = ConditionalTargetedTransfers(method=method, c_bar=2.15)

        if method == "density":
            tt.fit(X_train, y_train, r_train)
            for tol in condtol:
                tt.set_conditional_tolerance(tol)
                tt.run_opt(
                        X_test,
                        r_test,
                    )
                res = tt.evaluate(X_test, y_test, r_test)
                print('saving')
                write_result(save / f'{country}.csv', res, extra_run_labels)
                tt.evaluate_equity(
                    X_test,
                    y_test,
                    path=save / f'equity_{base_name}_undcondtol={tol}_condtol={tol}'
                )
        elif method == "qr":
            for tol in condtol:
                tt.set_conditional_tolerance(tol)
                tt.fit(X_train, y_train, r_train)
                tt.run_opt(
                        X_test,
                        r_test,
                    )
                res = tt.evaluate(X_test, y_test, r_test)
                write_result(save / f'{country}.csv', res, extra_run_labels)
                tt.evaluate_equity(
                    X_test,
                    y_test,
                    path=save / f'equity_{base_name}_uncondtol={tol}_condtol={tol}.csv'
                )

    elif constraint == "hybrid":

        tt = HybridTargetedTransfers(c_bar=2.15)
        tt.fit(X_train, y_train, r_train)
        
        for tol1 in uncondtol:
            for tol2 in condtol:
                tt.set_conditional_tolerance(tol2)
                tt.set_unconditional_tolerance(tol1)
                tt.run_opt(
                    X_test,
                    r_test,
                    path=save / f'{base_name}_uncondtol={tol1}_condtol={tol2}_opt.csv'
                )
                res = tt.evaluate(X_test, y_test, r_test)
                write_result(save / "{}.csv".format(country), res, extra_run_labels)
                tt.evaluate_equity(
                    X_test,
                    y_test,
                    path=save / f'equity_{base_name}_uncondtol={tol1}_condtol={tol2}.csv'
                )


# TODO: Update signature + file-writing
def run_with_fixed_transfer_amounts(
    country="malawi",
    d=2,
    tolerance=None,
    nclass=2,
    save="malawi_nclass_results.csv",
):
    X, y, r, features = get_dataset(country)

    (X_train, y_train, r_train), (X_test, y_test, r_test) = split_data(
        X=X[:, :d], y=y, r=None, p=0.6
    )  # for now not using sampling weights r

    tt = UnconditionalTargetedTransfers(name=country, tolerance=tolerance, c_bar=2.15)
    tt.fit(X_train, y_train, r_train)
    tt.run_opt(X_test, r_test, path="{}_d={}_tol={}.csv".format(country, d, tolerance))
    res = tt.evaluate(X_test, y_test, r_test)
    write_result(save + "{}.csv".format(country), res)

    dt = UnconditionalDiscreteTransfers(
        method="lindsey", nclass=None, c_bar=2.15, tolerance=tolerance
    )
    dt.fit(X_train, y_train, r_train)

    for n_c in nclass:
        dt.set_nclass(n_c)
        dt.run_opt(X_test, r_test)
        res = dt.evaluate(X_test, y_test, r_test)
        write_result(save + "{}.csv".format(country), res)



if __name__ == "__main__":
    pass
