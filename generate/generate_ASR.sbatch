#!/bin/bash

#SBATCH --job-name=mt_gen    # Job name
#SBATCH --output=/mnt/scratch-artemis/sonal/tower_speech/vllm_ASR.out    # Name of stdout output file (%j expands to %jobId)
#SBATCH --time=01:00:00         # Run time (hh:mm:ss) - 1.5 hours
#SBATCH --gres=gpu:1                # Number of GPUs to be used
#SBATCH --qos=gpu-short         # QOS to be used


declare -A path_dict=(
    ["covost"]="/mnt/scratch-artemis/kshitij/LLAMA/continued_pretraining/Continual_pretraining__covost_data/CoVoST_complete_test.json"
    ["ls_clean"]="/mnt/scratch-artemis/kshitij/LLAMA/continued_pretraining/librispeech_eval/after_pretraining/test-clean.json"
    ["ls_other"]="/mnt/scratch-artemis/kshitij/LLAMA/continued_pretraining/librispeech_eval/after_pretraining/test-other.json"
    ["fleurs"]="/mnt/scratch-artemis/sonal/10B_data/speech_data/fleurs_testset.jsonl"
    ["vox"]="/mnt/scratch-artemis/sonal/10B_data/speech_data/vox_testset.jsonl"
)

for data_id in "${!path_dict[@]}"; do
    path=${path_dict[$data_id]}
    python3 /mnt/scratch-artemis/sonal/tower_speech/evaluation/generate/generate_ASR.py "$data_id" "$path"
done


