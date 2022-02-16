#!/usr/bin/env bash
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -A sml
#SBATCH --exclude=yolanda

source activate py3.6

python -m experiment.run_experiment \
--model=${MODEL} \
--procfile=${PROC} \
--data=${DATA} \
--outdir=${OUT} \
--C=${C} \
--C_topics=${CTOPICS} \
--train_size=${TS} \
--epochs=30 \
--extra_epochs=10 \
--do_pretraining_stage \
--do_finetuning \
--train_test_mode
