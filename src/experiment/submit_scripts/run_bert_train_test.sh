#!/usr/bin/env bash
#SBATCH -A sml
#SBATCH -c 8
#SBATCH --gres=gpu:1

source activate py3.6

python -m experiment.run_bert \
--in_file=${PROC} \
--data=${DATA} \
--outdir=${OUT} \
--train_size=${TS} \
--train_test_mode