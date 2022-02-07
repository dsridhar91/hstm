#!/usr/bin/env bash
#SBATCH -A sml
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --mail-user=dhanya.sridhar@columbia.edu
#SBATCH --mail-type=ALL

source activate py3.6

python -m experiment.run_distillbert \
--in-file=${PROC} \
--data=${DATA} \
--outdir=${OUT} \
--split=${SPLIT}