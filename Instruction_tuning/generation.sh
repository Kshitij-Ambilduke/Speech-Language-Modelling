source /mnt/scratch-artemis/kshitij/LLAMA/instruction_tuning_codebase/env/insta_tune/bin/activate

python training_scripts/generation_asr.py \
    --base_model /mnt/scratch-artemis/kshitij/LLAMA/continued_pretraining/spgi_cont_eval/only_asr_intruct_tune/checkpoint-1506 \
    --lora_path /mnt/scratch-artemis/kshitij/LLAMA/continued_pretraining/TINY_LLAMA_OUTPUT/after_pretrain_after_insta \
    --data_path /mnt/scratch-artemis/kshitij/LLAMA/continued_pretraining/librispeech_eval/after_pretraining/test-clean.json \
    --output_path /mnt/scratch-artemis/kshitij/LLAMA/continued_pretraining/spgi_cont_eval/only_asr_intruct_tune/test-clean-onlyasr.txt \
    --batch_size 64 \
    --tokenizer_path /mnt/scratch-artemis/kshitij/LLAMA/Megatron_LLM/NONUNIFORM_1B_EXPERIMENT/Hf_ckpt/tinyllama-1b/4660 \
    --strategy 'greedy' \