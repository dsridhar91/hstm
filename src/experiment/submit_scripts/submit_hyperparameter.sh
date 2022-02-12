#!/usr/bin/env bash


export BASE_PROC=/proj/sml/projects/text-response/proc
export BASE_OUT=/proj/sml/projects/text-response/out
export BASE_LOG=/proj/sml/projects/text-response/log
export TOPIC=immigration
export BATCH_SIZE=512

# Experiment across first 4 settings

declare -a MODELS=(hstm-all)
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
				    export PRE=${BASE_PROC}/${DATAi}_pretraining.npz
				    export OUT=${BASE_OUT}/${DATAi}.lda_pretrained
				    LOG=${BASE_LOG}/${DATAi}.lda_pretrained
				    sbatch --job-name=hstm_hyper_pretr_${MODELi}_${DATAi}_exp_${Ci}.${CTOPICi}_${SPLITi} \
				           --output=${LOG}_${MODELi}_${Ci}.${CTOPICi}_${SPLITi}.out \
				           experiment/submit_scripts/run_experiment.sh
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
			    export PRE=${BASE_PROC}/${DATAi}_pretraining.npz
			    export OUT=${BASE_OUT}/${DATAi}.lda_pretrained
			    LOG=${BASE_LOG}/${DATAi}.lda_pretrained
			    sbatch --job-name=hstm_hyper_pretr_${MODELi}_${DATAi}_exp_${Ci}.${CTOPICi}_${SPLITi} \
			           --output=${LOG}_${MODELi}_${Ci}.${CTOPICi}_${SPLITi}.out \
			           experiment/submit_scripts/run_experiment.sh
			done
		done
	done
done

# Experiment across Media Framing Corpus 4 topics

export DATA=framing_corpus
export BATCH_SIZE=200

declare -a MODELS=(hstm-all)
declare -a TOPICS=(immigration deathpenalty samesex guncontrol)
declare -a CS=(1e-4 5e-5 1e-5 5e-6 1e-6)
declare -a CTS=(1e-4 5e-5 1e-5 5e-6 1e-6)


for MODELi in "${MODELS[@]}"; do 
	for TOPICi in "${TOPICS[@]}"; do
		for Ci in "${CS[@]}"; do
			for CTOPICi in "${CTS[@]}"; do
				for SPLITi in $(seq 0 4); do
					export MODEL=${MODELi}
					export TOPIC=${TOPICi}
				    export SPLIT=${SPLITi}
				    export C=${Ci}
				    export CTOPICS=${CTOPICi}
				    export PROC=${BASE_PROC}/${DATA}_${TOPICi}_proc.npz
				    export PRE=${BASE_PROC}/${DATA}_${TOPICi}_pretraining.npz
				    export OUT=${BASE_OUT}/${DATA}_${TOPICi}.lda_pretrained
				    LOG=${BASE_LOG}/${DATA}_${TOPICi}.lda_pretrained
				    sbatch --job-name=hstm_hyper_pretr_${MODELi}_${TOPICi}_exp_${Ci}.${CTOPICi}_${SPLITi} \
				           --output=${LOG}_${MODELi}_${Ci}.${CTOPICi}_${SPLITi}.out \
				           experiment/submit_scripts/run_experiment.sh
				done
			done
		done
	done
done

declare -a MODELS=(stm stm+bow)
declare -a TOPICS=(immigration deathpenalty samesex guncontrol)
declare -a CS=(1e-4 5e-5 1e-5 5e-6 1e-6)

for MODELi in "${MODELS[@]}"; do 
	for TOPICi in "${TOPICS[@]}"; do
		for Ci in "${CS[@]}"; do
			for SPLITi in $(seq 0 4); do
				export MODEL=${MODELi}
				export TOPIC=${TOPICi}
			    export SPLIT=${SPLITi}
			    export C=${Ci}
			    export PROC=${BASE_PROC}/${DATA}_${TOPICi}_proc.npz
			    export PRE=${BASE_PROC}/${DATA}_${TOPICi}_pretraining.npz
			    export OUT=${BASE_OUT}/${DATA}_${TOPICi}_.lda_pretrained
			    LOG=${BASE_LOG}/${DATA}_${TOPICi}.lda_pretrained
			    sbatch --job-name=hstm_hyper_pretr_${MODELi}_${TOPICi}_exp_${Ci}.${CTOPICi}_${SPLITi} \
			           --output=${LOG}_${MODELi}_${Ci}.${CTOPICi}_${SPLITi}.out \
			           experiment/submit_scripts/run_experiment.sh
			done
		done
	done
done
