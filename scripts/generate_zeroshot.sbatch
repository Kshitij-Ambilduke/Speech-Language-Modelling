#!/bin/bash

#SBATCH --job-name=tinyLoRA    # Job name
#SBATCH --output=/mnt/scratch-artemis/sonal/MT-experiments/slurm_outputs/eval_zero_wmt.out    # Name of stdout output file (%j expands to %jobId)
#SBATCH --time=4:00:00         # Run time (hh:mm:ss) - 1.5 hours
#SBATCH --gpus=1                # Number of GPUs to be used
#SBATCH --qos=gpu-short        # QOS to be used

source /mnt/scratch-artemis/kshitij/LLAMA/instruction_tuning_codebase/env/insta_tune/bin/activate

#CPT with text
python /mnt/scratch-artemis/sonal/MT-experiments/evaluation/scripts/generation_MT.py.save \
   --base_model /mnt/scratch-artemis/kshitij/LLAMA/Megatron_LLM/only_text_experiment/Hf_ckpt/tinyllama-1b/1000\
   --lora_path /mnt/scratch-artemis/kshitij/LLAMA/continued_pretraining/TINY_LLAMA_OUTPUT \
   --data_path /mnt/scratch-artemis/sonal/MT-experiments/evaluation/data/instruction_zero_shot_wmt.json \
   --output_path /mnt/scratch-artemis/sonal/MT-experiments/evaluation/outputs/cpt_text_wmt.txt \
   --batch_size 1 \
   --tokenizer_path /mnt/scratch-artemis/kshitij/LLAMA/Megatron_LLM/only_text_experiment/Hf_ckpt/tinyllama-1b \
   --strategy 'greedy'

#CPT with everything
python /mnt/scratch-artemis/sonal/MT-experiments/evaluation/scripts/generation_MT.py.save \
   --base_model /mnt/scratch-artemis/kshitij/LLAMA/Megatron_LLM/NONUNIFORM_1B_EXPERIMENT/Hf_ckpt/tinyllama-1b/4660 \
   --lora_path /mnt/scratch-artemis/kshitij/LLAMA/continued_pretraining/TINY_LLAMA_OUTPUT \
   --data_path /mnt/scratch-artemis/sonal/MT-experiments/evaluation/data/instruction_zero_shot_wmt.json \
   --output_path /mnt/scratch-artemis/sonal/MT-experiments/evaluation/outputs/cpt_no_text_wmt.txt \
   --batch_size 1 \
   --tokenizer_path /mnt/scratch-artemis/kshitij/LLAMA/Megatron_LLM/NONUNIFORM_1B_EXPERIMENT/Hf_ckpt/tinyllama-1b/4660 \
   --strategy 'greedy'

#CPT with text with extention
python /mnt/scratch-artemis/sonal/MT-experiments/evaluation/scripts/generation_MT.py.save \
   --base_model /mnt/scratch-artemis/kshitij/LLAMA/Megatron_LLM/only_text_experiment_with_extension/Hf_ckpt/tinyllama-1b/1000 \
   --lora_path /mnt/scratch-artemis/kshitij/LLAMA/continued_pretraining/TINY_LLAMA_OUTPUT \
   --data_path /mnt/scratch-artemis/sonal/MT-experiments/evaluation/data/instruction_zero_shot_wmt.json \
   --output_path /mnt/scratch-artemis/sonal/MT-experiments/evaluation/outputs/cpt_text_extension_wmt.txt \
   --batch_size 1 \
   --tokenizer_path /mnt/scratch-artemis/kshitij/LLAMA/Megatron_LLM/only_text_experiment_with_extension/Hf_ckpt/tinyllama-1b/1000 \
   --strategy 'greedy'
