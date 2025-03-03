#!/bin/bash
cd /home/selker/eop/poverty/dry_run
# No hyperparameter search needed
# Learning
python main_learn.py main --config /home/selker/eop/eop/run_dry_run/simulations/hparam/results/default_binary_gap.yaml --trainpath /data/eop/malawi/cleaned/train.parquet --testpath /data/eop/malawi/cleaned/test.parquet --summarypath /data/eop/malawi/cleaned/summary_2019.parquet --device cuda
sleep 1
python main_learn.py main --config /home/selker/eop/eop/run_dry_run/simulations/hparam/results/default_binary_rate.yaml --trainpath /data/eop/malawi/cleaned/train.parquet --testpath /data/eop/malawi/cleaned/test.parquet --summarypath /data/eop/malawi/cleaned/summary_2019.parquet --device cuda
sleep 1
python main_learn.py main --config /home/selker/eop/eop/run_dry_run/simulations/hparam/results/default_continuous_gap.yaml --trainpath /data/eop/malawi/cleaned/train.parquet --testpath /data/eop/malawi/cleaned/test.parquet --summarypath /data/eop/malawi/cleaned/summary_2019.parquet --device cuda
sleep 1
python main_learn.py main --config /home/selker/eop/eop/run_dry_run/simulations/hparam/results/default_continuous_rate.yaml --trainpath /data/eop/malawi/cleaned/train.parquet --testpath /data/eop/malawi/cleaned/test.parquet --summarypath /data/eop/malawi/cleaned/summary_2019.parquet --device cuda
sleep 1
