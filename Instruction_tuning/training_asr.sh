source /mnt/scratch-artemis/kshitij/LLAMA/instruction_tuning_codebase/env/insta_tune/bin/activate
include=localhost:$CUDA_VISIBLE_DEVICES
unset CUDA_VISIBLE_DEVICES

deepspeed --master_port 58000 --include=$include /mnt/scratch-artemis/kshitij/LLAMA/continued_pretraining/training_scripts/train_asr_after_pretrain.py \
    --model_name_or_path /mnt/scratch-artemis/kshitij/LLAMA/Megatron_LLM/NONUNIFORM_1B_EXPERIMENT/Hf_ckpt/tinyllama-1b/4660 \
    --data_path /mnt/scratch-artemis/kshitij/LLAMA/continued_pretraining/covost_translation_combined_cont/covost_traans600k.json \
    --trans_validation_data_path /mnt/scratch-artemis/kshitij/parallel_translation_data/DEV5k_instruction_tuning_translation_600k.json \
    --tokenizer_path /mnt/scratch-artemis/kshitij/LLAMA/Megatron_LLM/NONUNIFORM_1B_EXPERIMENT/Hf_ckpt/tinyllama-1b/4660 \
    --clean_validation_data_path /mnt/scratch-artemis/kshitij/LLAMA/continued_pretraining/TINY_LLAMA_OUTPUT/data_dev/CoVoST_complete_dev.json \
    --bf16 True \
    --output_dir /mnt/scratch-artemis/kshitij/LLAMA/continued_pretraining/spgi_cont_eval/instructune \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_total_limit 1 \
    --learning_rate 0.00008 \
    --save_strategy "epoch" \
    --group_by_length False \
    --weight_decay 0.0 \
    --warmup_ratio 0.02 \
    --warmup_steps 20 \
    --gradient_checkpointing False \
    --deepspeed ../config/ds_config_no_offload.json \
    --logging_steps 20 \
    --train_lora False \
    --lora_r 256 \
    --lora_alpha 512 \
    --lora_dropout 0.05 \
    --lora_target_modules "q_proj" "v_proj" "k_proj" \
    --complete_ft_modules "lm_head" "embed_tokens" \
    --lora_bias 'none' \
    --resume_from_checkpoint False \
    --do_eval False \
    --evaluation_strategy 'steps' \
    --per_device_eval_batch_size 1 \
    --eval_steps 500 \
    --report_to 'wandb'\
    --run_name '<< 4660 steps >>' \

