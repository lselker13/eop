#!/bin/bash
cd /home/selker/eop/poverty/dry_run
# Hyperparameter search
python main_hparam.py main --config /home/selker/eop/eop/run_dry_run/simulations/hparam/configs/gan_continuous_rate.yaml --learnsavedir /home/selker/eop/eop/run_dry_run/simulations/learn/results
sleep 1
python main_hparam.py main --config /home/selker/eop/eop/run_dry_run/simulations/hparam/configs/gan_binary_rate.yaml --learnsavedir /home/selker/eop/eop/run_dry_run/simulations/learn/results
sleep 1
python main_hparam.py main --config /home/selker/eop/eop/run_dry_run/simulations/hparam/configs/gan_binary_gap.yaml --learnsavedir /home/selker/eop/eop/run_dry_run/simulations/learn/results
sleep 1
python main_hparam.py main --config /home/selker/eop/eop/run_dry_run/simulations/hparam/configs/gan_continuous_gap.yaml --learnsavedir /home/selker/eop/eop/run_dry_run/simulations/learn/results
sleep 1
# Learning
python main_learn.py main --config /home/selker/eop/eop/run_dry_run/simulations/hparam/results/output_gan_continuous_rate.yaml --trainpath /data/eop/malawi/cleaned/train.parquet --testpath /data/eop/malawi/cleaned/test.parquet --summarypath /data/eop/malawi/cleaned/summary_2019.parquet --device cuda
sleep 1
python main_learn.py main --config /home/selker/eop/eop/run_dry_run/simulations/hparam/results/output_gan_binary_rate.yaml --trainpath /data/eop/malawi/cleaned/train.parquet --testpath /data/eop/malawi/cleaned/test.parquet --summarypath /data/eop/malawi/cleaned/summary_2019.parquet --device cuda
sleep 1
python main_learn.py main --config /home/selker/eop/eop/run_dry_run/simulations/hparam/results/output_gan_binary_gap.yaml --trainpath /data/eop/malawi/cleaned/train.parquet --testpath /data/eop/malawi/cleaned/test.parquet --summarypath /data/eop/malawi/cleaned/summary_2019.parquet --device cuda
sleep 1
python main_learn.py main --config /home/selker/eop/eop/run_dry_run/simulations/hparam/results/output_gan_continuous_gap.yaml --trainpath /data/eop/malawi/cleaned/train.parquet --testpath /data/eop/malawi/cleaned/test.parquet --summarypath /data/eop/malawi/cleaned/summary_2019.parquet --device cuda
sleep 1
