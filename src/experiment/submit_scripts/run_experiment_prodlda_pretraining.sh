#!/usr/bin/env bash
#SBATCH -A sml
#SBATCH -c 8
#SBATCH --mail-user=dhanya.sridhar@columbia.edu
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:1

source activate py3.6

python -m experiment.run_experiment \
--model=${MODEL} \
--pretraining_file=${PRE} \
--procfile=${PROC} \
--data=${DATA} \
--outdir=${OUT} \
--split=${SPLIT} \
--C=${C} \
--C_topics=${CTOPICS} \
--pretrained_prodlda \
--epochs=30 \
--do_finetuning \
--extra_epochs=10 \
--framing_topic=${TOPIC}