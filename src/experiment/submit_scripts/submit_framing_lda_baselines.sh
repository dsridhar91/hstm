#!/usr/bin/env bash


export BASE_PROC=/proj/sml/projects/text-response/proc
export BASE_OUT=/proj/sml/projects/text-response/out
export BASE_LOG=/proj/sml/projects/text-response/log
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
		# sbatch --job-name=pred_lda_${DATAi}_exp_${SPLITi} \
		# 	   --output=${LOG}_pred_lda_${SPLITi}.out \
		# 	   experiment/submit_scripts/run_lda_baseline.sh

		# sbatch --job-name=pred_lda_bow${DATAi}_exp_${SPLITi} \
		#    --output=${LOG}_pred_lda_bow${SPLITi}.out \
		#    experiment/submit_scripts/run_lda_bow_baseline.sh

		sbatch --job-name=slda_${DATA}_${TOPICi}_${SPLITi} \
	           --output=${LOG}_slda_${SPLITi}.out \
	           experiment/submit_scripts/run_slda.sh
	done
done
