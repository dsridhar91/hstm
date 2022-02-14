#!/usr/bin/env bash
#SBATCH -A sml
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --exclude=yolanda

source activate py3.6

python -m experiment.run_bert_sequence_class \
--in_file=${PROC} \
--data=${DATA} \
--outdir=${OUT} \
--split=${SPLIT}