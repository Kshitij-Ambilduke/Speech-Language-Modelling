from pathlib import Path
import sys
import evaluate
import pickle
from typing import Iterable, List, Union, Dict
import string
PathInput = Union[str, Path]
import deepspeed
from jsonargparse import CLI
import tqdm
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
# from evaluate import wer
import pandas as pd
import torch
from peft import PeftModel
import transformers
from transformers import LlamaForCausalLM, AutoTokenizer, AutoModelForCausalLM

from evaluate import load
bleu=load('bleu')
chrf=load('chrf')
#comet=load('comet')
IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "<PAD>"
DEFAULT_EOS_TOKEN = "\n"
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
    tokenizer_path: str = "/home/kshitij/LLAMA/instruction_tuning_codebase/TOKENIZERSS/tokenizer_used4training_2000clusters"
):
    """
    Generation script adapted from alpaca-lora
    (https://github.com/tloen/alpaca-lora/blob/main/generate.py)
    """
    torch.manual_seed(torch_seed)
    print(lora_path)
 
    prompts_df = pd.read_json(data_path)

    gt = prompts_df['output'].to_list()#[100:110]
    prompts = prompts_df["instruction"].to_list()#[100:110]

    if head is not None:
        prompts = prompts[:head]

    # prompts = torch.load("/home/kshitij/LLAMA/instruction_tuning_codebase/tokenized_test/test_norep.pt")

    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path , legacy=False, use_fast=True)
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=load_8bit,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    print(model)
    
    print("Length of tokenizer: ",len(tokenizer))


    # if True:
    #     print("LORAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAa")
    #     print("Loading LoRA checkpoint")
    #     model = PeftModel.from_pretrained(
    #         model,
    #         lora_path,
    #         torch_dtype=torch.bfloat16,
    #         device_map="auto",
    #     )
    #     print("Done!!!")

        
    # tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids('<PAD>')
    tokenizer.pad_token = '<PAD>'
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id


    # print(tokenizer.eos_token_id)
    print(tokenizer.eos_token)

    if not load_8bit:
        model.half()  # seems to fix bugs for some users.
        
    model.eval()
    # if torch.__version__ >= "2" and sys.platform != "win32":
    #     model = torch.compile(model)

    if strategy == "greedy":
        do_sample = False
    else:
        do_sample = True
    # batch prompts into list of lists of strings
    generated = []
    for i in tqdm.trange(0, len(prompts), batch_size):
    # for i in range(1):
        batch = prompts[i : i + batch_size]
        with torch.no_grad():
            inputs = tokenizer.batch_encode_plus(
                batch, padding="longest", return_tensors="pt",
                return_token_type_ids=None, truncation=True, max_length=2048
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
                max_length=2048
            )

        generated.extend(
            tokenizer.batch_decode(output[:, input_length:],skip_special_tokens=True)
        )
        # generated.extend(
        #     tokenizer.batch_decode(output,skip_special_tokens=False)
        # )
    # print(generated)
        # print(output)
        # print(attention_mask[:, input_length:])
        # print(output[:, input_length:])
        # print(output[:, input_length:].shape)
        # Only decode the part of the output that is new tokens
        # generated.extend(
        #     tokenizer.batch_decode(output[:, input_length:],skip_special_tokens=True)
        # )

    # save generated to file
    write_lines(Path(output_path), generated, escape_newline=True)
    # for i in range(len(generated)):
    #     print()
    #     print(gt[i])
    #     print(generated[i])
    #     print()
        
    
    # with open("/mnt/scratch-artemis/kshitij/LLAMA/continued_pretraining/TINY_LLAMA_OUTPUT/test.pkl", "wb") as fp:   #Pickling
    #     pickle.dump(generated, fp)
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


# if __name__ == "__main__":
#     CLI([generate], as_positional=False)


# if __name__ == "__main__":
#     generated, gt = CLI([generate], as_positional=False)
    # bleu = evaluate.load("bleu")
    # results = bleu.compute(predictions=generated, references=gt)

    # with open("OUTPUT_COVOST/OVERALL_RESULTS.txt",'a') as f:
    #     f.write(f"Translation results: {results}")
    #     f.write("\n")
    # print(results)
    # print("--------------------------")
if __name__ == "__main__":
    generated, gt, lora_path = CLI([generate], as_positional=False)

    for i in range(len(generated)):
        generated[i] = generated[i].strip().lower()
        generated[i] = generated[i].translate(str.maketrans ('', '', string.punctuation))

    for i in range(len(gt)):
        gt[i] = gt[i].strip().lower()
        gt[i] = gt[i].translate (str.maketrans('', '', string.punctuation))

    #results = comet.compute(source=generate,predictions=generated, references=gt)
    print("==================================================")
    #print(results)
    print("===============")
    bleu = evaluate.load("sacrebleu")
    results_bleu = bleu.compute(predictions=generated, references=gt)
    print(results_bleu)
    results_chrF = chrf.compute(predictions=generated, references=gt)
    print(results_chrF)
    with open("OVERALL_RESULTS.txt",'a') as f:
        f.write(f"COMET Lora Path: {lora_path} results: BLANK")
        f.write(f"\n BLEU results: {results_bleu} ChrF: {results_chrF}")
    # print("==================================")
    # with open("OUTPUT_COVOST/OVERALL_RESULTS.txt",'a') as f:
    #     f.write(f"Translation results: {results}")
    #     f.write("\n")

    # with open("OUTPUT_COVOST/OVERALL_RESULTS.txt","a") as f:
    #     f.write(f"ASR WER: {wer_clean} \n")
