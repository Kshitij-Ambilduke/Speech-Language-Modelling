#!/bin/bash

#SBATCH --job-name=tinyLoRA    # Job name
#SBATCH --output=/mnt/scratch-artemis/sonal/MT-experiments/slurm_outputs/eval_few_text.out    # Name of stdout output file (%j expands to %jobId)
#SBATCH --time=4:00:00         # Run time (hh:mm:ss) - 1.5 hours
#SBATCH --gres=gpu:1              # Number of GPUs to be used
#SBATCH --qos=gpu-short        # QOS to be used

source /mnt/scratch-artemis/kshitij/LLAMA/instruction_tuning_codebase/env/insta_tune/bin/activate

# five shot output on text only CPT model
python /mnt/scratch-artemis/sonal/MT-experiments/evaluation/generation_MT.py.save \
   --base_model /mnt/scratch-artemis/kshitij/LLAMA/Megatron_LLM/only_text_experiment/Hf_ckpt/tinyllama-1b/1000\
   --lora_path /mnt/scratch-artemis/kshitij/LLAMA/continued_pretraining/TINY_LLAMA_OUTPUT \
   --data_path /mnt/scratch-artemis/sonal/MT-experiments/evaluation/data/instruction_few_shot_wmt.json \
   --output_path /mnt/scratch-artemis/sonal/MT-experiments/test_outputs/textonly_wmt_few.txt \
   --batch_size 1 \
   --tokenizer_path /mnt/scratch-artemis/kshitij/LLAMA/Megatron_LLM/only_text_experiment/Hf_ckpt/tinyllama-1b/1000 \
   --strategy 'greedy'
