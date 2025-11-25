from pathlib import Path
import os
import numpy as np
import shutil
import yaml


C_BAR = 3.0
BUDGETS = np.linspace(0.05, 3.0, 15)
ORACLE_BUDGETS = np.linspace(0.0, 3.0, 15)

scripts_directory = Path('/home/selker/eop/eop/run_togo_simulations/scripts')

top_level_simulations_directory = Path('/home/selker/eop/eop/run_togo_simulations/simulations')
dry_run_code_path = Path('/home/selker/eop/poverty/dry_run')

def generate_runs(data_path, country, clear_existing_configs=False):
    data_path = Path(data_path)

    if data_path.name == country:
        setting_code = country
    else:
        setting_code = country + '_' + data_path.name

    trainpath = f"{data_path}/train.parquet"
    testpath = f"{data_path}/test.parquet"
    summarypath = f"{data_path}/summary.parquet"
    hparam_savedir = top_level_simulations_directory / 'hparam' / 'results' / setting_code
    base_config = {
        "savedir": f"{hparam_savedir}",
        "device": 'cuda',
        "data": {
            "geo_extrapolation": True,
            "outcome": "consumption_per_capita_per_day",
            "weight": "headcount_adjusted_hh_wgt",
            "gt": {
                "trainpath": trainpath,
                "summarypath": summarypath
            },
        },
    }

    default_nn_config = {
        "n_regressors": [10, 20, 30],
        "neural_network": {
            "n_layers": [1, 2, 3],
            "n_hidden_units": [256, 1024, 2048],
            "lr": [0.001, 0.005, 0.01],
        },
    }

    binary_gap_config = base_config.copy()
    binary_gap_config["binary_gap"] = default_nn_config.copy()

    binary_rate_config = base_config.copy()
    binary_rate_config["binary_rate"] = default_nn_config.copy()

    continuous_gap_config = base_config.copy()
    continuous_gap_config["continuous_gap"] = default_nn_config.copy()

    continuous_rate_config = base_config.copy()

    continuous_rate_config["continuous_rate"] = {
        "n_alpha": [50, 100, 200],
        "density_estimation": {
            "n_features": [5, 10, 15, 20],
            "n_bins": [10, 50, 100, 200],
            "n_knots": [2, 6, 12],
            "degree": [2, 4, 6],
            "kde_fft": True if country in ["colombia", "india"] else False,
            "winsorize": True if country in ["colombia", "india"] else False,
        },
    }

    modern_pmt_config = base_config.copy()
    modern_pmt_config["modern_pmt"] = default_nn_config.copy()
    del modern_pmt_config["modern_pmt"]["n_regressors"]

    pmt_config = base_config.copy()
    pmt_config["pmt"] = {
        "transfer_value": C_BAR,
        "lasso": {"alpha": [0, 0.01, 0.1, 1.0]},
    }

    learn_savedir = top_level_simulations_directory / 'learn' / 'results' / setting_code

    oracle_config = {
        "oracle_gap": {},
        "data": {
            "outcome": "consumption_per_capita_per_day",
            "weight": "headcount_adjusted_hh_wgt",
            "geo_extrapolation": True,
        },
        "savedir": f"{learn_savedir}",
    }
    
    ubi_config = {
        "ubi": {},
        "data": {
            "outcome": "consumption_per_capita_per_day",
            "weight": "headcount_adjusted_hh_wgt",
            "geo_extrapolation": True,
        },
        "savedir": f"{learn_savedir}",
    }

    # Create necessary directories

    if clear_existing_configs:
        shutil.rmtree(top_level_simulations_directory / 'learn' / 'configs' / setting_code, ignore_errors=True)
        shutil.rmtree(top_level_simulations_directory / 'hparam' / 'configs' / setting_code, ignore_errors=True)

    (top_level_simulations_directory / 'learn' / 'configs' / setting_code).mkdir(parents=True, exist_ok=True)
    (top_level_simulations_directory / 'learn' / 'results' / setting_code).mkdir(parents=True, exist_ok=True)

    (top_level_simulations_directory / 'hparam' / 'configs' / setting_code).mkdir(parents=True, exist_ok=True)
    (top_level_simulations_directory / 'hparam' / 'results' / setting_code).mkdir(parents=True, exist_ok=True)

    names = [
        # "gt_continuous_rate",
        # "gt_binary_rate",
        # "gt_binary_gap",
        "gt_continuous_gap",
        # "gt_modern_pmt",
        # "gt_pmt",
    ]
    configs = [
        # continuous_rate_config,
        # binary_rate_config,
        # binary_gap_config,
        continuous_gap_config,
        # modern_pmt_config,
        # pmt_config,
    ]
    for i, name in enumerate(names):
        config = configs[i]
        with (top_level_simulations_directory / "hparam" / "configs" / setting_code / f"{name}.yaml").open('w') as file:
            yaml.dump(config, file, default_flow_style=False)

    with (top_level_simulations_directory / "learn" / "configs" / setting_code / f"oracle_gap.yaml").open('w') as file:
        yaml.dump(oracle_config, file, default_flow_style=False)

    with (top_level_simulations_directory / "learn" / "configs" / setting_code / f"ubi.yaml").open('w') as file:
        yaml.dump(ubi_config, file, default_flow_style=False)

    script_file_path = scripts_directory / f"{setting_code}_script.sh"

    with script_file_path.open('w') as f:
        print("#!/bin/bash", file=f)
        print(f"cd {dry_run_code_path}", file=f)
        print("# Hyperparameter search", file=f)
        for config in (top_level_simulations_directory / 'hparam' / 'configs' / setting_code).iterdir():

            base_cmd = (
                f"python main_hparam.py main "
                f"--config {config} "
                f"--learnsavedir {top_level_simulations_directory}/learn/results/{setting_code} "
            )
            print(base_cmd, file=f)
            print("sleep 1", file=f)

        print("# Learning", file=f)
        for config in (top_level_simulations_directory / 'hparam' / 'configs' / setting_code).iterdir():
            hparam_output_config_path = f"{top_level_simulations_directory}/hparam/results/{setting_code}/output_{config.name}"

            base_cmd = (
                f"python main_learn.py main "
                f"--config {hparam_output_config_path} "
                f"--trainpath {trainpath} "
                f"--testpath {testpath} "
                f"--summarypath {summarypath} "
                f"--device cuda "
                f"--country {country} "
                f"--povertyline 2.15 "
                f"--year 2017 "
            )
            
            print(base_cmd, file=f)
            print("sleep 1", file=f)
        
        # Oracle and ubi, directly referencing fixed learn config
        for config in (top_level_simulations_directory / "learn" / "configs" / setting_code).iterdir():

            base_cmd = (
                f"python main_learn.py main "
                f"--config {config} "
                f"--trainpath {trainpath} "
                f"--testpath {testpath} "
                f"--summarypath {summarypath} "
                f"--device cuda "
                f"--country {country} "
                f"--povertyline 2.15 "
                f"--year 2017 "
            )
        
            print(base_cmd, file=f)
            print("sleep 1", file=f)

    os.chmod(script_file_path, 0o755) 


if __name__ == '__main__':

    scripts_directory.mkdir(exist_ok=True)
    data_path = '/data/eop/country_data/togo/cleaned/satellite'
    generate_runs(data_path, 'togo')

    data_path = '/data/eop/country_data/togo/cleaned/survey_satellite_sample'
    generate_runs(data_path, 'togo')