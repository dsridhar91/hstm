#!/usr/bin/env bash


export BASE_PROC=/proj/sml/projects/text-response/proc
export BASE_OUT=/proj/sml/projects/text-response/out
export BASE_LOG=/proj/sml/projects/text-response/log

declare -a DATASETS=(amazon peerread semantic_scholar yelp amazon_binary)


for DATAi in "${DATASETS[@]}"; do	
	for SPLITi in $(seq 0 4); do
		export DATA=${DATAi}
		export SPLIT=${SPLITi}
		export PROC=${BASE_PROC}/${DATAi}_proc.npz
		export OUT=${BASE_OUT}/${DATAi}
		LOG=${BASE_LOG}/${DATAi}
		sbatch --job-name=pred_lda_${DATAi}_exp_${SPLITi} \
			   --output=${LOG}_pred_lda_${SPLITi}.out \
			   experiment/submit_scripts/run_lda_baseline.sh

		sbatch --job-name=pred_lda_bow${DATAi}_exp_${SPLITi} \
		   --output=${LOG}_pred_lda_bow${SPLITi}.out \
		   experiment/submit_scripts/run_lda_bow_baseline.sh

		sbatch --job-name=slda_${DATAi}_exp_${SPLITi} \
	           --output=${LOG}_slda_${SPLITi}.out \
	           experiment/submit_scripts/run_slda.sh
	done
done
