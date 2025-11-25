#!/bin/bash
cd /home/selker/eop/poverty/dry_run
# Hyperparameter search
python main_hparam.py main --config /home/selker/eop/eop/run_togo_simulations/simulations/hparam/configs/togo_survey_satellite_sample/gt_continuous_gap.yaml --learnsavedir /home/selker/eop/eop/run_togo_simulations/simulations/learn/results/togo_survey_satellite_sample 
sleep 1
# Learning
python main_learn.py main --config /home/selker/eop/eop/run_togo_simulations/simulations/hparam/results/togo_survey_satellite_sample/output_gt_continuous_gap.yaml --trainpath /data/eop/country_data/togo/cleaned/survey_satellite_sample/train.parquet --testpath /data/eop/country_data/togo/cleaned/survey_satellite_sample/test.parquet --summarypath /data/eop/country_data/togo/cleaned/survey_satellite_sample/summary.parquet --device cuda --country togo --povertyline 2.15 --year 2017 
sleep 1
python main_learn.py main --config /home/selker/eop/eop/run_togo_simulations/simulations/learn/configs/togo_survey_satellite_sample/oracle_gap.yaml --trainpath /data/eop/country_data/togo/cleaned/survey_satellite_sample/train.parquet --testpath /data/eop/country_data/togo/cleaned/survey_satellite_sample/test.parquet --summarypath /data/eop/country_data/togo/cleaned/survey_satellite_sample/summary.parquet --device cuda --country togo --povertyline 2.15 --year 2017 
sleep 1
python main_learn.py main --config /home/selker/eop/eop/run_togo_simulations/simulations/learn/configs/togo_survey_satellite_sample/ubi.yaml --trainpath /data/eop/country_data/togo/cleaned/survey_satellite_sample/train.parquet --testpath /data/eop/country_data/togo/cleaned/survey_satellite_sample/test.parquet --summarypath /data/eop/country_data/togo/cleaned/survey_satellite_sample/summary.parquet --device cuda --country togo --povertyline 2.15 --year 2017 
sleep 1
