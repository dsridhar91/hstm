#!/usr/bin/env bash


export BASE_PROC=/proj/sml/projects/text-response/proc
export BASE_OUT=/proj/sml/projects/text-response/out
export BASE_LOG=/proj/sml/projects/text-response/log
export MODEL=slda

declare -a DATASETS=(amazon peerread yelp amazon_binary)

for DATAi in "${DATASETS[@]}"; do	
	for SPLITi in $(seq 0 4); do
		export DATA=${DATAi}
		export SPLIT=${SPLITi}
		export PROC=${BASE_PROC}/${DATAi}_proc.npz
		export OUT=${BASE_OUT}/${DATAi}
		LOG=${BASE_LOG}/${DATAi}
		
		sbatch --job-name=slda_${DATAi}_exp_${SPLITi} \
	           --output=${LOG}_slda_${SPLITi}.out \
	           experiment/submit_scripts/run_experiment.sh
	done
done

export DATA=framing_corpus
export BATCH_SIZE=200

declare -a TOPICS=(immigration deathpenalty samesex guncontrol)

for TOPICi in "${TOPICS[@]}"; do	
	for SPLITi in $(seq 0 4); do
		export TOPIC=${TOPICi}
		export SPLIT=${SPLITi}
		export PROC=${BASE_PROC}/${DATA}_${TOPICi}_proc.npz
		export OUT=${BASE_OUT}/${DATA}_${TOPICi}
		LOG=${BASE_LOG}/${DATA}_${TOPICi}

		sbatch --job-name=slda_${DATA}_${TOPICi}_${SPLITi} \
	           --output=${LOG}_slda_${SPLITi}.out \
	           experiment/submit_scripts/run_experiment.sh
	done
done