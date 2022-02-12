#!/usr/bin/env bash

export TOPIC=immigration
export PROC=/proj/sml/projects/text-response/proc/yelp_full_proc.npz
export OUT=/proj/sml/projects/text-response/out/yelp_full
export LOG=/proj/sml/projects/text-response/log/yelp_full
export BATCH_SIZE=512
export DATA=yelp_full
export TS=560000

export MODEL=hstm-all
export C=1e-5
export CTOPICS=1e-6
export SPLIT=0

sbatch --job-name=yelp_full_${MODEL}\
       --output=${LOG}_${MODEL}.out \
       experiment/submit_scripts/run_train_test_experiment.sh

