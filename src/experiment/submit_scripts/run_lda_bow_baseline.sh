#!/usr/bin/env bash
#SBATCH -A sml
#SBATCH -c 8
#SBATCH --mail-user=dhanya.sridhar@columbia.edu
#SBATCH --mail-type=ALL

source activate py3.6

python -m experiment.run_lda_baselines \
--procfile=${PROC} \
--data=${DATA} \
--outdir=${OUT} \
--split=${SPLIT} \
--predict_with_bow