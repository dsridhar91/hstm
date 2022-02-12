#!/usr/bin/env bash

export TOPIC=immigration
export PROC=/proj/sml/projects/text-response/proc/yelp_full_proc.npz
export OUT=/proj/sml/projects/text-response/out/yelp_full
export LOG=/proj/sml/projects/text-response/log/yelp_full
export BATCH_SIZE=512
export DATA=yelp_full
export TS=560000


export MODEL=hstm-all
export C=
export CTOPICS=

for SPLITi in $(seq 0 4); do
    export SPLIT=${SPLITi}
    sbatch --job-name=yelp_benchmark_${MODEL}_${SPLITi} \
           --output=${LOG}_${MODEL}_${SPLITi}.out \
           experiment/submit_scripts/run_train_test_experiment.sh
done
