#!/bin/bash
cd /home/selker/eop/poverty/dry_run
# Hyperparameter search
python main_hparam.py main --config /home/selker/eop/eop/run_togo_simulations/simulations/hparam/configs/togo_satellite_and_survey/gt_continuous_gap.yaml --learnsavedir /home/selker/eop/eop/run_togo_simulations/simulations/learn/results/togo_satellite_and_survey 
sleep 1
python main_hparam.py main --config /home/selker/eop/eop/run_togo_simulations/simulations/hparam/configs/togo_satellite_and_survey/gt_modern_pmt.yaml --learnsavedir /home/selker/eop/eop/run_togo_simulations/simulations/learn/results/togo_satellite_and_survey 
sleep 1
# Learning
python main_learn.py main --config /home/selker/eop/eop/run_togo_simulations/simulations/hparam/results/togo_satellite_and_survey/output_gt_continuous_gap.yaml --trainpath /data/eop/country_data/togo/cleaned/satellite_and_survey/train.parquet --testpath /data/eop/country_data/togo/cleaned/satellite_and_survey/test.parquet --summarypath /data/eop/country_data/togo/cleaned/satellite_and_survey/summary.parquet --device cuda --country togo --povertyline 2.15 --year 2017 
sleep 1
python main_learn.py main --config /home/selker/eop/eop/run_togo_simulations/simulations/hparam/results/togo_satellite_and_survey/output_gt_modern_pmt.yaml --trainpath /data/eop/country_data/togo/cleaned/satellite_and_survey/train.parquet --testpath /data/eop/country_data/togo/cleaned/satellite_and_survey/test.parquet --summarypath /data/eop/country_data/togo/cleaned/satellite_and_survey/summary.parquet --device cuda --country togo --povertyline 2.15 --year 2017 
sleep 1
python main_learn.py main --config /home/selker/eop/eop/run_togo_simulations/simulations/learn/configs/togo_satellite_and_survey/oracle_gap.yaml --trainpath /data/eop/country_data/togo/cleaned/satellite_and_survey/train.parquet --testpath /data/eop/country_data/togo/cleaned/satellite_and_survey/test.parquet --summarypath /data/eop/country_data/togo/cleaned/satellite_and_survey/summary.parquet --device cuda --country togo --povertyline 2.15 --year 2017 
sleep 1
python main_learn.py main --config /home/selker/eop/eop/run_togo_simulations/simulations/learn/configs/togo_satellite_and_survey/ubi.yaml --trainpath /data/eop/country_data/togo/cleaned/satellite_and_survey/train.parquet --testpath /data/eop/country_data/togo/cleaned/satellite_and_survey/test.parquet --summarypath /data/eop/country_data/togo/cleaned/satellite_and_survey/summary.parquet --device cuda --country togo --povertyline 2.15 --year 2017 
sleep 1
