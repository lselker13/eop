#!/bin/bash
cd /home/selker/eop/poverty/dry_run
# Training WGAN
python main_gan.py train --device cpu --savedir /home/selker/eop/eop/run_dry_run/simulations/gan/pickled --trainpath /data/eop/malawi/cleaned/train.parquet --summarypath /data/eop/malawi/cleaned/summary_2019.parquet --maxepochs 2500 --lr 0.001 --batchsize 128 --dropout 0.0
sleep 1
# Generating synthetic data
python main_gan.py generate --objectspath /home/selker/eop/eop/run_dry_run/simulations/gan/pickled/objects-maxepochs=2500_lr=0.001_batchsize=128_dropout=0.0_trial=0.pickle --nsamples 20000 --savedir /home/selker/eop/eop/run_dry_run/simulations/data
sleep 1
