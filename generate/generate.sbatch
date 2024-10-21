#!/bin/bash

#SBATCH --job-name=mt_gen    # Job name
#SBATCH --output=/mnt/scratch-artemis/sonal/tower_speech/vllm_eval.out    # Name of stdout output file (%j expands to %jobId)
#SBATCH --time=01:00:00         # Run time (hh:mm:ss) - 1.5 hours
#SBATCH --gres=gpu:1                # Number of GPUs to be used
#SBATCH --qos=gpu-short         # QOS to be used


languages=('de' 'es' 'fr' 'it' 'ko' 'nl' 'pt' 'ru' 'zh')

# Loop through each language identifier and run generate.py with it
for lang in "${languages[@]}"; do
    python3 /mnt/scratch-artemis/sonal/tower_speech/evaluation/generate/generate.py "$lang"
done


