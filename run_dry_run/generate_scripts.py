from pathlib import Path
import os

scripts_directory = Path('scripts')

top_level_simulations_directory = Path('/home/selker/eop/eop/run_dry_run/simulations')
data_path = Path('/data/eop/malawi/cleaned')
dry_run_code_path = Path('/home/selker/eop/poverty/dry_run')

def generate_gt_runs():
    # Create necessary directories
    (top_level_simulations_directory / 'learn').mkdir(parents=True, exist_ok=True)
    (top_level_simulations_directory / 'hparam').mkdir(parents=True, exist_ok=True)

    hparam_configs = [
        "gt_continuous_rate.yaml",
        "gt_binary_rate.yaml",
        "gt_binary_gap.yaml",
        "gt_continuous_gap.yaml",
    ]

    learn_configs = [
        "output_gt_continuous_rate.yaml",
        "output_gt_binary_rate.yaml",
        "output_gt_binary_gap.yaml",
        "output_gt_continuous_gap.yaml",
    ]

    script_file_path = scripts_directory / "gt_script.sh"
    with script_file_path.open('w') as f:
        print("#!/bin/bash", file=f)
        print(f"cd {dry_run_code_path}", file=f)
        print("# Hyperparameter search", file=f)
        for config in hparam_configs:
            base_cmd = (
                f"python main_hparam.py main "
                f"--config {top_level_simulations_directory}/hparam/configs/{config} "
                f"--learnsavedir {top_level_simulations_directory}/learn/results"
            )
            print(base_cmd, file=f)
            print("sleep 1", file=f)

        print("# Learning", file=f)
        for config in learn_configs:
            base_cmd = (
                f"python main_learn.py main "
                f"--config {top_level_simulations_directory}/hparam/results/{config} "
                f"--trainpath {data_path}/train.parquet "
                f"--testpath {data_path}/test.parquet "
                f"--summarypath {data_path}/summary_2019.parquet "
                f"--device cuda"
            )
            print(base_cmd, file=f)
            print("sleep 1", file=f)

    os.chmod(script_file_path, 0o755) 


def generate_default_runs():
    # Create necessary directories
    (top_level_simulations_directory / 'learn').mkdir(parents=True, exist_ok=True)

    learn_configs = [
        "default_binary_gap.yaml",
        "default_binary_rate.yaml",
        "default_continuous_gap.yaml",
        "default_continuous_rate.yaml",
    ]
    script_file_path = scripts_directory / "default_script.sh"
    with script_file_path.open('w') as f:
        print("#!/bin/bash", file=f)
        print(f"cd {dry_run_code_path}", file=f)

        print("# No hyperparameter search needed", file=f)

        print("# Learning", file=f)
        for config in learn_configs:
            base_cmd = (
                f"python main_learn.py main "
                f"--config {top_level_simulations_directory}/hparam/results/{config} "
                f"--trainpath {data_path}/train.parquet "
                f"--testpath {data_path}/test.parquet "
                f"--summarypath {data_path}/summary_2019.parquet "
                f"--device cuda"
            )
            print(base_cmd, file=f)
            print("sleep 1", file=f)
    
    os.chmod(script_file_path, 0o755) 


def generate_oracle_runs():
    # Create necessary directories
    (top_level_simulations_directory / 'learn').mkdir(parents=True, exist_ok=True)

    learn_configs = [
        "oracle_gap.yaml",
        "oracle_rate.yaml"
    ]
    script_file_path = scripts_directory / "oracle_script.sh"
    with script_file_path.open('w') as f:
        print("#!/bin/bash", file=f)
        print(f"cd {dry_run_code_path}", file=f)

        print("# No hyperparameter search needed", file=f)

        print("# Learning", file=f)
        for config in learn_configs:
            base_cmd = (
                f"python main_learn.py main "
                f"--config {top_level_simulations_directory}/hparam/results/{config} "
                f"--trainpath {data_path}/train.parquet "
                f"--testpath {data_path}/test.parquet "
                f"--summarypath {data_path}/summary_2019.parquet "
                f"--device cuda"
            )
            print(base_cmd, file=f)
            print("sleep 1", file=f)
    
    os.chmod(script_file_path, 0o755) 


def generate_train_wgan_and_generate_synthetic_data_runs():
    # Create necessary directories
    (top_level_simulations_directory / 'gan/pickled').mkdir(parents=True, exist_ok=True)
    (top_level_simulations_directory / 'data').mkdir(parents=True, exist_ok=True)

    maxepochs = 2500
    lr = 1e-3
    batchsize = 128
    dropout = 0.0

    script_base_name = (
        f"wgan_maxepochs={maxepochs}_lr={lr}_batchsize={batchsize}_dropout={dropout}"
    )

    script_file_path = scripts_directory / f'{script_base_name}.sh'

    with script_file_path.open('w') as f:
        print("#!/bin/bash", file=f)
        print(f"cd {dry_run_code_path}", file=f)
        print('# Training WGAN', file=f)
        base_cmd = (
            f"python main_gan.py train "
            f"--device cpu "
            f"--savedir {top_level_simulations_directory}/gan/pickled "
            f"--trainpath {data_path}/train.parquet "
            f"--summarypath {data_path}/summary_2019.parquet "
            f"--maxepochs {maxepochs} "
            f"--lr {lr} "
            f"--batchsize {batchsize} "
            f"--dropout {dropout}"
        )
        print(base_cmd, file=f)

        print('sleep 1', file=f)

        print('# Generating synthetic data', file=f)

        base_cmd = (
            f"python main_gan.py generate "
            f"--objectspath {top_level_simulations_directory}/gan/pickled/objects-maxepochs={maxepochs}_lr={lr}_batchsize={batchsize}_dropout={dropout}_trial=0.pickle "
            f"--nsamples 20000 "
            f"--savedir {top_level_simulations_directory}/data"
        )
        print(base_cmd, file=f)
        print("sleep 1", file=f)
    
    os.chmod(script_file_path, 0o755) 



def generate_gan_hparam_and_learning_runs():

    # Create necessary directories
    (top_level_simulations_directory / 'learn').mkdir(parents=True, exist_ok=True)
    (top_level_simulations_directory / 'hparam').mkdir(parents=True, exist_ok=True)

    hparam_configs = [
        "gan_continuous_rate.yaml",
        "gan_binary_rate.yaml",
        "gan_binary_gap.yaml",
        "gan_continuous_gap.yaml",
    ]

    learn_configs = [
        "output_gan_continuous_rate.yaml",
        "output_gan_binary_rate.yaml",
        "output_gan_binary_gap.yaml",
        "output_gan_continuous_gap.yaml",
    ]

    script_file_path = scripts_directory / "gan_script.sh"
    with script_file_path.open('w') as f:
        print("#!/bin/bash", file=f)
        print(f"cd {dry_run_code_path}", file=f)
        print("# Hyperparameter search", file=f)
        for config in hparam_configs:
            base_cmd = (
                f"python main_hparam.py main "
                f"--config {top_level_simulations_directory}/hparam/configs/{config} "
                f"--learnsavedir {top_level_simulations_directory}/learn/results"
            )
            print(base_cmd, file=f)
            print("sleep 1", file=f)

        print("# Learning", file=f)
        for config in learn_configs:
            base_cmd = (
                f"python main_learn.py main "
                f"--config {top_level_simulations_directory}/hparam/results/{config} "
                f"--trainpath {data_path}/train.parquet "
                f"--testpath {data_path}/test.parquet "
                f"--summarypath {data_path}/summary_2019.parquet "
                f"--device cuda"
            )
            print(base_cmd, file=f)
            print("sleep 1", file=f)

    os.chmod(script_file_path, 0o755) 
    


if __name__ == '__main__':

    generate_gt_runs()
    generate_default_runs()
    generate_train_wgan_and_generate_synthetic_data_runs()
    generate_gan_hparam_and_learning_runs()
    generate_oracle_runs()