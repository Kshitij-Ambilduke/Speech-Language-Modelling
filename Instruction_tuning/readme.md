# Preliminaries

Clone the repo and run:

```bash
cd Speech-Language-Modelling/Instruction_tuning
python -m venv <venv_path>
source <venv_path>/bin/activate
pip install -r requirements.txt
```

# Preparing data

The data passed to either the training bash script, or the generation script, must be a json file with the following structure:

```json
[
    {
        "instruction": <instruction_in_plain_text_1>,
        "output": <output_in_plain_text_1>
    },

    {
        "instruction": <instruction_in_plain_text_2>,
        "output": <output_in_plain_text_2>
    },

    .
    .
    .
]
```

# Training Scripts

## Configs

We are using the training bash scripts as configuration files.

The training script accepts arguments that the `transformers.HfArgumentParser` accepts, so you can pass them in the training bash as well. We lay out a few examples here. 

> If you wish to pass arguments related to the optimizer and learning rate scheduler, please remove those lines from the deepspeed config! (e.g., lines 5 to 20 in `../config/ds_config_no_offload.json`).

`training_asr.sh` contains a template for a training bash script. Here's the description of its arguments:

  - `model_name_or_path`: path to pretrained hf model.
  - `data_path`: path to the json file which have the data that you want to use. It should be a json file with only two attributes "instruction" and "output".
  - `trans_validation_data_path` : path to the json file containing translation validation data, Again this data has to be in a json file in the form of list-of-dictionaries with keys "instruction" and "output". 
  - `tokenizer_path` : path to the tokenizer. For instruction tuning on just text, tokenizer path can be set same as the model path.
  - `clean_validation_data_path` : Path to ASR validation data.
  - `bf16`: whether you want to use bfloat16 or not.
  - `output_dir`: directory where is going to be save the preprocessed data and the finetuned models or the adapters.
  - `num_train_epochs`: total number of training epochs.
  - `per_device_train_batch_size`: batch size per device.
  - `gradient_accumulation_steps`: Can be used to increase the effective batch size (= num_gpus * batch_size_per_gpu * gradient_accumulation_steps)
  - `evaluation_strategy`:  the evaluation strategy to adopt during training. As we are not using validation set, it should be "no".
  - `save_total_limit`: If a value is passed, will limit the total amount of checkpoints. Deletes the older checkpoints in output_dir.
  - `learning_rate`: learning rate.
  - `save_strategy`: For now it should be "epoch".
  - `group_by_length`:
  - `report_to`: If you want to monitor your training you can select "tensorboard" or "wandb".
  - `weight_decay`:
  - `warmup_ratio`:
  - `gradient_checkpointing`: 
  - `deepspeed`: path to the deepspeed configuration file.
  - `logging_steps`: print training loss and learning rate every _logging_steps_.

Lora specific parameters:
  - `train_lora`: whether you want to train LoRAs or perform full finetuning.
  - `lora_r`: Lora attention dimension.
  - `lora_alpha`: The alpha parameter for Lora scaling.
  - `lora_dropout`: he dropout probability for Lora layers.
  - `lora_target_modules`: The names of the modules to apply Lora to. depending on the model that you are using, the names are different. See [this](https://github.com/huggingface/peft/blob/86290e9660d24ef0d0cedcf57710da249dd1f2f4/src/peft/utils/other.py) script for a few examples (line 220 onwards).
  - `complete_ft_modules`: The names of modules to finetune completely without insertion of Lora adapters. 

For checking the name of usual Lora modules of different LLMs check [this out](https://github.com/huggingface/peft/blob/86290e9660d24ef0d0cedcf57710da249dd1f2f4/src/peft/utils/other.py#L220). Besides these, other [huggingface trainer](https://huggingface.co/docs/transformers/en/main_classes/trainer#transformers.TrainingArguments) arguments can be passed as required.

About deepspeed configs
`deepspeed_config` folder contains DeepSpeed configuration files for different setups. The available configurations are:

* `ds_config_no_offload.json` : Configuration that does not offload optimizer nor parameters. It provides faster performance.

* `ds_config_offload_optimizer.json` : Configuration that offloads optimizer states to CPUs. It allows for larger models or batch sizes, although it may be slightly slower.

* `ds_config_offload_optimizer_param.json` : Configuration that offloads both optimizer states and model parameters to the CPU. It is the slowest option but can be useful if you encounter CUDA out of memory issues.



---

# Generation Scripts

To use the generation script, run the following bash command:

```bash
python scripts/generation.py \
  --base_model <path_to_hf_model> \
  --lora_path <path_to_lora> \
  --data_path <path_to_dataset> \
  --batch_size <batch_size> \
  --output_path <path_to_output> \
  --strategy <strategy> \
  --temperature <temperature> \
  --p <p> \
  --top_k <top_k> \
  --num_beams <num_beams> \
  --max_new_tokens <max_new_tokens> \
  --torch_seed <torch_seed> \
  --head <head> \
  --load_8bit <load_8bit> \
  --return_output <return_output>
```

- The data used here should have the same format as in training.
- Leave `--lora_path` empty if your model was not finetuned using LoRA.
- The parameters below `output_path` pertain to the generation process. 
- If using `--strategy greedy` (the default), the parameters until `--num_beams` do not matter. Pass `--strategy None` to change this. 
- `--head` determines the number of observations of the dataset generation should be performed over (e.g., passing 10 will only consider the first 10 rows of the data). Useful for debugging.
- `--load_8bit` should be used if the model was finetuned with 8bit.

An example is given in `generation.sh`