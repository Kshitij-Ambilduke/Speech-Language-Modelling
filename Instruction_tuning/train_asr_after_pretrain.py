#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li

#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at

#        http://www.apache.org/licenses/LICENSE-2.0

#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import copy
import io
import json
import logging
import os
import pickle
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import deepspeed
import torch
import transformers
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
from peft import get_peft_model, get_peft_model_state_dict, LoraConfig, PeftModel, prepare_model_for_int8_training
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import Trainer


os.environ["WANDB_PROJECT"] = "TinyLLaMA-continued-pretraining"
os.environ["WANDB_ENTITY"] = "hajmola"

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    train_lora: bool = field(default=True)
    lora_r: int = field(default=8)
    lora_alpha: int = field(default=16)
    lora_target_modules: List[str] = field(default_factory=list)
    lora_dropout: float = field(default=0.05)
    lora_bias: str = field(default="none")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    clean_validation_data_path: str = field(default=None, metadata={"help": "Path to the Validation data."})
    tokenizer_path: str = field(
        default="/home/kshitij/LLAMA/model_weights/huggingface", metadata={"help": "Path to tokenizer"}
    )
    trans_validation_data_path: str = field(default=None, metadata={"help": "Path to the Validation data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    train_8bit: bool = field(default=False)
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:

    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    # print(input_ids[0].dtype)
    return dict(input_ids=input_ids, labels=labels)


def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    jdict = json.load(f)
    f.close()
    return jdict


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = jload(data_path)

        logging.warning("Formatting inputs...")
        sources = [f"{example['instruction']} " for example in list_data_dict]
        targets = [f"{example['output']} {tokenizer.eos_token}" for example in list_data_dict]

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        attention_mask[:, -1] = True  # PAD = EOS, so we do this so that final token is considered for attention
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path)

    clean_eval_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.clean_validation_data_path)

    # trans_eval_dataset = SupervisedDataset(
    #     tokenizer=tokenizer, data_path=data_args.trans_validation_data_path
    # )

    # eval_dataset = {'ASR_Eval':clean_eval_dataset,'MT_Eval':trans_eval_dataset}

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=clean_eval_dataset, data_collator=data_collator)


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        use_cache=False if training_args.gradient_checkpointing else True,
    )
    # print(model_args.lora_target_modules)
    if training_args.train_8bit:
        model = prepare_model_for_int8_training(model)

    print("Loading Tokenizer")
    tokenizer = transformers.AutoTokenizer.from_pretrained(data_args.tokenizer_path, legacy=False)

    print(len(tokenizer))

    if model_args.train_lora:
        print(model_args.lora_target_modules)
        print("~~~~~~~~~~~~~~~~~~~~~~~~ Training With LoRA ~~~~~~~~~~~~~~~~~~~~~~~~")
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        if (
            os.path.isfile(training_args.output_dir + "/adapter_model.bin")
            and training_args.resume_from_checkpoint == "True"
        ):
            print("~~~~~~~~~~~~~~~~~~~~~~~~ Loading LoRA Checkpoint ~~~~~~~~~~~~~~~~~~~~~~~~")
            model = PeftModel.from_pretrained(model, training_args.output_dir, is_trainable=True)
            model.print_trainable_parameters()

        else:
            print("~~~~~~~~~~~~~~~~~~~~~~~~ Loading New LoRA ~~~~~~~~~~~~~~~~~~~~~~~~")
            config = LoraConfig(
                r=model_args.lora_r,
                lora_alpha=model_args.lora_alpha,
                target_modules=model_args.lora_target_modules,
                lora_dropout=model_args.lora_dropout,
                bias=model_args.lora_bias,
                task_type="CAUSAL_LM",
                modules_to_save=["lm_head", "embed_tokens"],
            )

            model = get_peft_model(model, config)
            model.print_trainable_parameters()

    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("<PAD>")
    tokenizer.pad_token = "<PAD>"
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    print("Vocab size= ", len(tokenizer))
    print("Padding token ID: ", tokenizer.pad_token_id)
    print("Padding token: ", tokenizer.pad_token)

    data_path = os.path.join(training_args.output_dir, "data")
    if os.path.isfile(data_path):
        with open(data_path, "rb") as f:
            data_module = pickle.load(f)
    else:
        data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
        if not os.path.exists(training_args.output_dir):
            os.mkdir(training_args.output_dir)
        with open(data_path, "wb") as f:
            pickle.dump(data_module, f)

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=data_module["train_dataset"],
        eval_dataset=data_module["eval_dataset"],
        data_collator=data_module["data_collator"],
    )

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    # prompt = "English speech: <extra_id_38><extra_id_3814><extra_id_3154><extra_id_139><extra_id_1741><extra_id_294><extra_id_1003><extra_id_4002><extra_id_214><extra_id_1003><extra_id_774><extra_id_2085><extra_id_531><extra_id_2281><extra_id_2789><extra_id_2525><extra_id_1354><extra_id_2046><extra_id_2393><extra_id_4333><extra_id_3142><extra_id_4472><extra_id_3353><extra_id_3976><extra_id_3786><extra_id_4599><extra_id_3355><extra_id_1569><extra_id_1345><extra_id_2668><extra_id_4553><extra_id_1543><extra_id_709><extra_id_1796><extra_id_2851><extra_id_1873><extra_id_3485><extra_id_3896><extra_id_3800><extra_id_2559><extra_id_4941><extra_id_1795><extra_id_2268><extra_id_1186><extra_id_3163><extra_id_1696><extra_id_2965><extra_id_1531><extra_id_2307><extra_id_4571><extra_id_2330><extra_id_1><extra_id_2559><extra_id_1045><extra_id_187><extra_id_4758><extra_id_1986><extra_id_463><extra_id_4768><extra_id_3918><extra_id_2197><extra_id_273><extra_id_4077><extra_id_4599><extra_id_2186><extra_id_3355><extra_id_1569><extra_id_1345><extra_id_1851><extra_id_4553><extra_id_1543><extra_id_709><extra_id_1796><extra_id_3477><extra_id_195><extra_id_391><extra_id_4620><extra_id_272><extra_id_627><extra_id_555><extra_id_4276><extra_id_3098><extra_id_962><extra_id_2250><extra_id_4710><extra_id_531><extra_id_4710><extra_id_4974><extra_id_637><extra_id_4974><extra_id_248><extra_id_3306><extra_id_2587><extra_id_877><extra_id_1089><extra_id_77><extra_id_643><extra_id_877><extra_id_643><extra_id_946><extra_id_4366><extra_id_3367><extra_id_1867> \n German text:"
    # inputs = tokenizer.encode_plus(
    #             prompt, padding="longest", return_tensors="pt",
    #             return_token_type_ids=None,
    #         )
    # input_length = inputs.input_ids.shape[1]
    # output = model.generate(
    #             **inputs,
    #             max_new_tokens=500)
    # print(output)
    # print(tokenizer.batch_decode(output[:, input_length:],skip_special_tokens=True))
    # quit()

    # trainer.train('/mnt/scratch-artemis/kshitij/LLAMA/continued_pretraining/spgi_cont_eval/only_asr_intruct_tune/one-epoch-checkpoint-1130')
    trainer.train()

    if model_args.train_lora:
        model.save_pretrained(training_args.output_dir)
        folders = next(os.walk(training_args.output_dir))[1]

        matching_folders = [folder for folder in folders if folder.startswith("checkpoint-")]

        match = re.search(r"\d+", matching_folders[0])
        number = match.group()

        model.to("cpu")
        if torch.distributed.get_rank() == 0:
            state_dict = get_fp32_state_dict_from_zero_checkpoint(
                training_args.output_dir,
                tag="checkpoint-" + str(number) + "/global_step" + str(number),
            )
            print(len(state_dict))
            d = get_peft_model_state_dict(model, state_dict=state_dict)
            # print(d)
            torch.save(d, training_args.output_dir + "/adapter_model.bin")
            # shutil.rmtree(training_args.output_dir + "/checkpoint-" + str(number))
        else:
            pass


if __name__ == "__main__":
    train()
