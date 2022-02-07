#!/usr/bin/env bash


export BASE_PROC=/proj/sml/projects/text-response/proc
export BASE_OUT=/proj/sml/projects/text-response/out
export BASE_LOG=/proj/sml/projects/text-response/log
export DATA=framing_corpus
export BATCH_SIZE=300
export MODEL=hstm-all

# export TOPIC=immigration
# export C=1e-5
# export CTOPICS=1e-6
# export PROC=${BASE_PROC}/${DATA}_${TOPIC}_proc.npz
# export PRE=${BASE_PROC}/${DATA}_${TOPIC}_pretraining.npz
# export OUT=${BASE_OUT}/${DATA}_${TOPIC}.model_save

# for SPLITi in $(seq 0 4); do
#     export SPLIT=${SPLITi}
#     export MODEL_FILE=${BASE_OUT}/${DATA}_${TOPIC}_${SPLITi}.model
#     LOG=${BASE_LOG}/hstm_save_${DATA}_${TOPIC}
#     sbatch --job-name=hstm_model_save_${MODEL}_${TOPIC}_${SPLITi} \
#            --output=${LOG}_${SPLITi}.out \
#            experiment/submit_scripts/run_save.sh
# done


export TOPIC=samesex
export C=5e-5
export CTOPICS=1e-6
export PROC=${BASE_PROC}/${DATA}_${TOPIC}_proc.npz
export PRE=${BASE_PROC}/${DATA}_${TOPIC}_pretraining.npz
export OUT=${BASE_OUT}/${DATA}_${TOPIC}.model_save

for SPLITi in $(seq 0 4); do
    export SPLIT=${SPLITi}
    export MODEL_FILE=${BASE_OUT}/${DATA}_${TOPIC}_${SPLITi}.model
    LOG=${BASE_LOG}/hstm_save_${DATA}_${TOPIC}
    sbatch --job-name=hstm_model_save_${MODEL}_${TOPIC}_${SPLITi} \
           --output=${LOG}_${SPLITi}.out \
           experiment/submit_scripts/run_save.sh
done

# export TOPIC=guncontrol
# export C=5e-5
# export CTOPICS=1e-6
# export PROC=${BASE_PROC}/${DATA}_${TOPIC}_proc.npz
# export PRE=${BASE_PROC}/${DATA}_${TOPIC}_pretraining.npz
# export OUT=${BASE_OUT}/${DATA}_${TOPIC}.model_save

# for SPLITi in $(seq 0 4); do
#     export SPLIT=${SPLITi}
#     export MODEL_FILE=${BASE_OUT}/${DATA}_${TOPIC}_${SPLITi}.model
#     LOG=${BASE_LOG}/hstm_save_${DATA}_${TOPIC}
#     sbatch --job-name=hstm_model_save_${MODEL}_${TOPIC}_${SPLITi} \
#            --output=${LOG}_${SPLITi}.out \
#            experiment/submit_scripts/run_save.sh
# done

# export TOPIC=deathpenalty
# export C=5e-6
# export CTOPICS=5e-6
# export PROC=${BASE_PROC}/${DATA}_${TOPIC}_proc.npz
# export PRE=${BASE_PROC}/${DATA}_${TOPIC}_pretraining.npz
# export OUT=${BASE_OUT}/${DATA}_${TOPIC}.model_save

# for SPLITi in $(seq 0 4); do
#     export SPLIT=${SPLITi}
#     export MODEL_FILE=${BASE_OUT}/${DATA}_${TOPIC}_${SPLITi}.model
#     LOG=${BASE_LOG}/hstm_save_${DATA}_${TOPIC}
#     sbatch --job-name=hstm_model_save_${MODEL}_${TOPIC}_${SPLITi} \
#            --output=${LOG}_${SPLITi}.out \
#            experiment/submit_scripts/run_save.sh
# done
