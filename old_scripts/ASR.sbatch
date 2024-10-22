#!/bin/bash

#SBATCH --job-name=vllm    # Job name
#SBATCH --output=/mnt/scratch-artemis/sonal/vllm_ASR.out    # Name of stdout output file (%j expands to %jobId)
#SBATCH --time=01:00:00         # Run time (hh:mm:ss) - 1.5 hours
#SBATCH --gres=gpu:1                # Number of GPUs to be used
#SBATCH --qos=gpu-short         # QOS to be used

python3 /mnt/scratch-artemis/sonal/MT-experiments/vllm_ASR.py
