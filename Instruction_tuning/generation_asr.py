import sys
from pathlib import Path
from typing import Dict, Iterable, List, Union


PathInput = Union[str, Path]
import pandas as pd
import torch
import tqdm
import transformers
from evaluate import load
from jsonargparse import CLI
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from whisper_normalizer.english import EnglishTextNormalizer


wer = load("wer")

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "<PAD>"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


def generate(
    base_model: str = "",
    lora_path: str = "",
    data_path: str = "",
    output_path: str = "",
    batch_size: int = 1,
    strategy: str = "greedy",
    temperature: float = 1.0,
    p: float = 1.0,
    top_k: int = 1,
    num_beams: int = 1,
    max_new_tokens: int = 256,
    torch_seed: int = 123,
    head: int = None,
    load_8bit: bool = False,
    return_output: bool = True,
    tokenizer_path: str = "/home/kshitij/LLAMA/instruction_tuning_codebase/TOKENIZERSS/tokenizer_used4training_2000clusters",
):
    """
    Generation script adapted from alpaca-lora
    (https://github.com/tloen/alpaca-lora/blob/main/generate.py)
    """
    torch.manual_seed(torch_seed)
    print(lora_path)

    prompts_df = pd.read_json(data_path)

    gt = prompts_df["output"].to_list()  # [0:110]
    prompts = prompts_df["instruction"].to_list()  # [0:110]

    if head is not None:
        prompts = prompts[:head]

    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path, legacy=False, use_fast=True)

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=load_8bit,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    print(model)

    print("Length of tokenizer: ", len(tokenizer))

    if lora_path != "":
        print("Loading LoRA checkpoint")
        model = PeftModel.from_pretrained(
            model,
            lora_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        print("Done!!!")

    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    print(tokenizer.eos_token)

    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    if strategy == "greedy":
        do_sample = False
    else:
        do_sample = True
    # batch prompts into list of lists of strings
    generated = []
    for i in tqdm.trange(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        with torch.no_grad():
            inputs = tokenizer.batch_encode_plus(
                batch,
                padding="longest",
                return_tensors="pt",
                return_token_type_ids=None,
            ).to("cuda")
            input_length = inputs.input_ids.shape[1]
            output = model.generate(
                **inputs,
                do_sample=do_sample,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=p,
                top_k=top_k,
                num_beams=num_beams,
                # min_new_tokens=2
            )

        generated.extend(tokenizer.batch_decode(output[:, input_length:], skip_special_tokens=True))

    write_lines(Path(output_path), generated, escape_newline=True)
    return generated, gt, lora_path


def write_lines(
    path: PathInput,
    lines: Iterable[str],
    escape_newline: bool = False,
) -> None:
    """Writes lines to a file.

    Lines can be escaped, meaning \n is transformed to \\n.

    Args:
        path: The path to the file.
        lines: The lines to write.
        escape_newline: Whether to escape newlines.
    """
    # make dir, if not exists
    path.parent.mkdir(parents=True, exist_ok=True)
    if escape_newline:
        lines = (l.replace("\n", "\\n") for l in lines)
    with open(path, "w") as f:
        f.writelines((f"{l}\n" for l in lines))


if __name__ == "__main__":
    generated, gt, lora_path = CLI([generate], as_positional=False)
    english_normalizer = EnglishTextNormalizer()

    for i in range(len(generated)):
        generated[i] = generated[i].strip()
        generated[i] = english_normalizer(generated[i])  # .translate(str.maketrans ('', '', string.punctuation))

    for i in range(len(gt)):
        gt[i] = gt[i].strip()
        gt[i] = english_normalizer(gt[i])  # .translate (str.maketrans('', '', string.punctuation))

    results = wer.compute(predictions=generated, references=gt)
    print("==================================================")
    print(results)
    print("==================================================")
    print(lora_path)
    print(results)
