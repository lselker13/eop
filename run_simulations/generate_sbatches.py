import itertools
import glob
import os


SBATCH_PREFACE = """#!/bin/bash
#SBATCH -t 12:00:00
#SBATCH -p normal
#SBATCH -c 1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name="{}.sh"
#SBATCH --error="{}/{}_err.log"
#SBATCH --output="{}/{}_out.log"\n
"""

# constants for commands

OUTPUT_PATH = "/Users/leo/Documents/gpl/eop/poverty/replication/scripts/"
SAVE_PATH = "/Users/leo/Documents/gpl/eop/poverty/replication/results/"


def generate_malawi_n_class_runs():
    ds = [0, 3, 12, 20]
    for d in ds:
        exp_id = "malawi_nclass_d={}".format(d)
        script_fn = os.path.join(OUTPUT_PATH, "{}.sh".format(exp_id))
        with open(script_fn, "w") as f:
            print(
                SBATCH_PREFACE.format(exp_id, OUTPUT_PATH, exp_id, OUTPUT_PATH, exp_id),
                file=f,
            )
            base_cmd = "python fixed_transfer_amt.py main --d {} --country malawi --tolerance 0.10 --nclass 2 3 5 8 10 15 20 --save {}".format(
                d, SAVE_PATH
            )
            print(base_cmd, file=f)
            print("sleep 1", file=f)


def generate_malawi_runs():
    ds = [0, 3, 12, 20]

    constraints = ["unconditional", "conditional"]
    quantile_methods = ["qr", "density"]

    for d in ds:
        for constraint in constraints:
            if constraint == "conditional":
                for quantile_method in quantile_methods:
                    exp_id = "malawi_d={}_constraint={}_method={}".format(
                        d, constraint, quantile_method
                    )
                    script_fn = os.path.join(OUTPUT_PATH, "{}.sh".format(exp_id))
                    with open(script_fn, "w") as f:
                        print(
                            SBATCH_PREFACE.format(
                                exp_id, OUTPUT_PATH, exp_id, OUTPUT_PATH, exp_id
                            ),
                            file=f,
                        )
                        base_cmd = "python main.py main --d {} --constraint {} --method {} --country malawi --tolerance 0.05 0.075 0.10 0.15 0.20 --save {}".format(
                            d, constraint, quantile_method, SAVE_PATH
                        )
                        print(base_cmd, file=f)
                        print("sleep 1", file=f)

            elif constraint == "unconditional":
                exp_id = "malawi_d={}_constraint={}".format(d, constraint)
                script_fn = os.path.join(OUTPUT_PATH, "{}.sh".format(exp_id))
                with open(script_fn, "w") as f:
                    print(
                        SBATCH_PREFACE.format(
                            exp_id, OUTPUT_PATH, exp_id, OUTPUT_PATH, exp_id
                        ),
                        file=f,
                    )
                    base_cmd = "python main.py main --d {} --constraint {} --country malawi --tolerance 0.05 0.075 0.10 0.15 0.20 --save {}".format(
                        d, constraint, SAVE_PATH
                    )
                    print(base_cmd, file=f)
                    print("sleep 1", file=f)


# generate_sim_runs()

# generate_1d_runs()
# generate_uganda_runs()
generate_malawi_runs()
