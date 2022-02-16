#!/usr/bin/env bash
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -A sml
#SBATCH --exclude=yolanda

source activate py3.6

python -m experiment.run_experiment \
--model=${MODEL} \
--pretraining_file=${PRE} \
--model_file=${MODEL_FILE} \
--procfile=${PROC} \
--data=${DATA} \
--outdir=${OUT} \
--split=${SPLIT} \
--C=${C} \
--C_topics=${CTOPICS} \
--pretrained \
--epochs=30 \
--do_finetuning \
--extra_epochs=10 \
--framing_topic=${TOPIC} \
--batch_size=${BATCH_SIZE} \
--save