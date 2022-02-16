#!/usr/bin/env bash
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -A sml
#SBATCH --exclude=yolanda

source activate py3.6

python -m experiment.run_bert \
--in_file=${PROC} \
--data=${DATA} \
--outdir=${OUT} \
--split=${SPLIT}