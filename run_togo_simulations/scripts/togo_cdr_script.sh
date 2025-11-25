#!/bin/bash
cd /home/selker/eop/poverty/dry_run
# Hyperparameter search
python main_hparam.py main --config /home/selker/eop/eop/run_togo_simulations/simulations/hparam/configs/togo_cdr/gt_continuous_gap.yaml --learnsavedir /home/selker/eop/eop/run_togo_simulations/simulations/learn/results/togo_cdr 
sleep 1
python main_hparam.py main --config /home/selker/eop/eop/run_togo_simulations/simulations/hparam/configs/togo_cdr/gt_modern_pmt.yaml --learnsavedir /home/selker/eop/eop/run_togo_simulations/simulations/learn/results/togo_cdr 
sleep 1
# Learning
python main_learn.py main --config /home/selker/eop/eop/run_togo_simulations/simulations/hparam/results/togo_cdr/output_gt_continuous_gap.yaml --trainpath /data/eop/country_data/togo/cleaned/cdr/train.parquet --testpath /data/eop/country_data/togo/cleaned/cdr/test.parquet --summarypath /data/eop/country_data/togo/cleaned/cdr/summary.parquet --device cuda --country togo --povertyline 2.15 --year 2017 
sleep 1
python main_learn.py main --config /home/selker/eop/eop/run_togo_simulations/simulations/hparam/results/togo_cdr/output_gt_modern_pmt.yaml --trainpath /data/eop/country_data/togo/cleaned/cdr/train.parquet --testpath /data/eop/country_data/togo/cleaned/cdr/test.parquet --summarypath /data/eop/country_data/togo/cleaned/cdr/summary.parquet --device cuda --country togo --povertyline 2.15 --year 2017 
sleep 1
python main_learn.py main --config /home/selker/eop/eop/run_togo_simulations/simulations/learn/configs/togo_cdr/oracle_gap.yaml --trainpath /data/eop/country_data/togo/cleaned/cdr/train.parquet --testpath /data/eop/country_data/togo/cleaned/cdr/test.parquet --summarypath /data/eop/country_data/togo/cleaned/cdr/summary.parquet --device cuda --country togo --povertyline 2.15 --year 2017 
sleep 1
python main_learn.py main --config /home/selker/eop/eop/run_togo_simulations/simulations/learn/configs/togo_cdr/ubi.yaml --trainpath /data/eop/country_data/togo/cleaned/cdr/train.parquet --testpath /data/eop/country_data/togo/cleaned/cdr/test.parquet --summarypath /data/eop/country_data/togo/cleaned/cdr/summary.parquet --device cuda --country togo --povertyline 2.15 --year 2017 
sleep 1
