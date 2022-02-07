#!/usr/bin/env bash
#SBATCH -A sml
#SBATCH --exclude=yolanda
#SBATCH -c 8
#SBATCH --mail-user=dhanya.sridhar@columbia.edu
#SBATCH --mail-type=ALL

source activate py3.6

python -m experiment.run_lda_baselines \
--predict_with_z \
--procfile=${PROC} \
--data=${DATA} \
--framing_topic=${TOPIC} \
--outdir=${OUT} \
--split=${SPLIT} \
--batch_size=${BATCH_SIZE}