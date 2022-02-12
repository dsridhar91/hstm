#!/usr/bin/env bash


export PROC=/proj/sml/projects/text-response/csv_proc/yelp_full.csv
export OUT=/proj/sml/projects/text-response/out/yelp_full
export LOG=/proj/sml/projects/text-response/log/yelp_full
export DATA=yelp_full
export TS=560000
export SPLIT=0
  
sbatch --job-name=yelp_full_bert \
       --output=${LOG}.bert.out \
       experiment/submit_scripts/run_bert_train_test.sh



