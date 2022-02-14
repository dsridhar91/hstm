#!/usr/bin/env bash


export BASE_PROC=/proj/sml/projects/text-response/csv_proc
export BASE_OUT=/proj/sml/projects/text-response/out
export BASE_LOG=/proj/sml/projects/text-response/log

declare -a DATASETS=(amazon peerread amazon_binary yelp immigration samesex deathpenalty guncontrol)

for DATAi in "${DATASETS[@]}"; do
	for SPLITi in $(seq 0 4); do
		export DATA=${DATAi}
	    export SPLIT=${SPLITi}
	    export PROC=${BASE_PROC}/${DATAi}.csv
	    export OUT=${BASE_OUT}/${DATAi}
	    LOG=${BASE_LOG}/${DATAi}
	    sbatch --job-name=bert_seq.${DATAi}.${SPLITi} \
	           --output=${LOG}.bert_seq.${SPLITi}.out \
	           experiment/submit_scripts/run_bert_seq_class.sh
	done

done

