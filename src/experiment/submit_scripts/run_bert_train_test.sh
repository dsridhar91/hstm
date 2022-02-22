#!/usr/bin/env bash

#SBATCH -A sml
#SBATCH --exclude=yolanda

source activate py3.6

python -m experiment.run_bert \
--in_file=${PROC} \
--data=${DATA} \
--outdir=${OUT} \
--train_size=${TS} \
--batch_size=16 \
--epochs=3 \
--train_test_mode