from evaluate import load
from vllm import LLM, SamplingParams
import pandas as pd
import json
from whisper_normalizer.basic import BasicTextNormalizer
import jsonlines
import sys

data = sys.argv[1]
data_path = sys.argv[2]
normalizer = BasicTextNormalizer()
#tokenizer_path = "/mnt/data-artemis/duarte/tower_speech/tinyllama_10B_hf_ckpts/tinyllama-1b/4500"
tokenizer_path="/mnt/scratch-artemis/sonal/tower_speech/data/tokenizer/"
sampling_params=SamplingParams(stop=['<\s>','\n', "\\n"], max_tokens=128, temperature=0) #,max-num-seqs==1)
#model_path = "/mnt/data-artemis/duarte/tower_speech/tower-base-6B-5to1/llama2-7b/1500"
#model_path ="/mnt/data-artemis/duarte/tower_speech/tower-base-8B-5to3/1500"
#model_path="/mnt/data-artemis/duarte/tower_speech/tower-base-6B-5to1/llama2-7b/1500"
model_path="/mnt/data-artemis/duarte/tower_speech/tower-base-6B-5to1/llama2-7b/3000"
#model_path="/mnt/data-artemis/duarte/tower_speech/tower-base-6B-5to1/llama2-7b/5000"
#model_path="/mnt/data-artemis/duarte/tower_speech/tower-base-8B-5to3/llama2-7b/3000"
#model_path="/mnt/data-artemis/duarte/tower_speech/ls-test/llama2-7b/1500"
#model_path="/mnt/data-artemis/duarte/tower_speech/ls-test-3k-steps-hf-ckpts/llama2-7b/3000"
out_root = "/mnt/scratch-artemis/sonal/tower_speech/evaluation/results/ASR/"
#prepare into prompt list
def read_data(data_path):
    prompts_df = pd.read_json(data_path)
    ref = prompts_df["output"].to_list()
    prompts = prompts_df["instruction"].to_list()
    return [prompts[i].strip(" ").strip("\n").replace("English speech: ","Speech: ").replace("English text:","English: ") for i in range(len(prompts))], [normalizer(ref[i]) for i in range(len(ref))]

def vllm_generate(out_file, model_path, prompts, ref):
    generated  = []
    llm=LLM(model=model_path, tokenizer=tokenizer_path,tokenizer_mode="slow")
    outputs = llm.generate(prompts, sampling_params)
    for output in outputs:
        prompt = output.prompt
        print(prompt)
        generated_text = output.outputs[0].text
        generated.append(normalizer(generated_text))
        
        out_file.write(normalizer(generated_text))
        out_file.write("\n")
        
    #compute metric
    wer= load('wer')

    results_wer = wer.compute(predictions=generated, references=ref)
    results = {'wer':results_wer}
    print(results)
    return results, generated


wers = []

f = open(out_root + "ASR_check_{}.txt".format(data),'w')
prompts, ref= read_data(data_path)
results, generated = vllm_generate(f, model_path, prompts, ref)
f.write("Results wer: {} ".format(results['wer']))
f.write("\n")
