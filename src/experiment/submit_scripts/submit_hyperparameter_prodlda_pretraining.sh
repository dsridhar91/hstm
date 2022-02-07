#!/usr/bin/env bash


export BASE_PROC=/proj/sml/projects/text-response/proc
export BASE_OUT=/proj/sml/projects/text-response/out
export BASE_LOG=/proj/sml/projects/text-response/log
export TOPIC=immigration
export BATCH_SIZE=512

declare -a MODELS=(hstm hstm-nobeta hstm-all)
declare -a DATASETS=(amazon peerread yelp amazon_binary)
declare -a CS=(1e-4 5e-5 1e-5 5e-6 1e-6)
declare -a CTS=(1e-4 5e-5 1e-5 5e-6 1e-6)


for MODELi in "${MODELS[@]}"; do 
	for DATAi in "${DATASETS[@]}"; do
		for Ci in "${CS[@]}"; do
			for CTOPICi in "${CTS[@]}"; do
				for SPLITi in $(seq 0 4); do
					export MODEL=${MODELi}
					export DATA=${DATAi}
				    export SPLIT=${SPLITi}
				    export C=${Ci}
				    export CTOPICS=${CTOPICi}
				    export PROC=${BASE_PROC}/${DATAi}_proc.npz
				    export PRE=${BASE_PROC}/${DATAi}_prodlda_pretraining.npz
				    export OUT=${BASE_OUT}/${DATAi}.prodlda_pretrained
				    LOG=${BASE_LOG}/${DATAi}.prodlda_pretrained
				    sbatch --job-name=hstm_hyper_prodlda_pretr_${MODELi}_${DATAi}_exp_${Ci}.${CTOPICi}_${SPLITi} \
				           --output=${LOG}_${MODELi}_${Ci}.${CTOPICi}_${SPLITi}.out \
				           experiment/submit_scripts/run_experiment_prodlda_pretraining.sh
				done
			done
		done
	done
done

declare -a MODELS=(stm stm+bow)
declare -a DATASETS=(amazon peerread yelp amazon_binary)
declare -a CS=(1e-4 5e-5 1e-5 5e-6 1e-6)

for MODELi in "${MODELS[@]}"; do 
	for DATAi in "${DATASETS[@]}"; do
		for Ci in "${CS[@]}"; do
			for SPLITi in $(seq 0 4); do
				export MODEL=${MODELi}
				export DATA=${DATAi}
			    export SPLIT=${SPLITi}
			    export C=${Ci}
			    export PROC=${BASE_PROC}/${DATAi}_proc.npz
			    export PRE=${BASE_PROC}/${DATAi}_prodlda_pretraining.npz
			    export OUT=${BASE_OUT}/${DATAi}.prodlda_pretrained
			    LOG=${BASE_LOG}/${DATAi}.prodlda_pretrained
			    sbatch --job-name=hstm_hyper_prodlda_pretr_${MODELi}_${DATAi}_exp_${Ci}.${CTOPICi}_${SPLITi} \
			           --output=${LOG}_${MODELi}_${Ci}.${CTOPICi}_${SPLITi}.out \
			           experiment/submit_scripts/run_experiment_prodlda_pretraining.sh
			done
		done
	done
done
