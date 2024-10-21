#!/bin/bash

#SBATCH --job-name=vllm    # Job name
#SBATCH --output=/mnt/scratch-artemis/sonal/vllm_ASR.out    # Name of stdout output file (%j expands to %jobId)
#SBATCH --time=01:00:00         # Run time (hh:mm:ss) - 1.5 hours
#SBATCH --gres=gpu:1                # Number of GPUs to be used
#SBATCH --qos=gpu-short         # QOS to be used

python3 /mnt/scratch-artemis/sonal/tower_speech/evaluation/generate/ASR_Test.py --token_path "/mnt/scratch-artemis/sonal/tower_speech/data/tokenizer" --model_path "/mnt/data-artemis/duarte/tower_speech/ls-test-3k-steps-hf-ckpts/llama2-7b/3000" --data_path "/mnt/scratch-artemis/kshitij/LLAMA/continued_pretraining/librispeech_eval/after_pretraining/test-clean.json"
