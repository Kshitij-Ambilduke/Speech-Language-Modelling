from evaluate import load
from vllm import LLM, SamplingParams
import pandas as pd
import json
import jsonlines
from collections import defaultdict
import sys
langs = ['de','es','fr','it','ko','nl','pt','ru','zh']

lang = sys.argv[1]
lang_pair =  "{}-en".format(lang)
lang_dict = {'de':"German",'es':"Spanish",'fr':"French","it":"Italian","ko":"Korean","nl":"Dutch","pt":"Portuguese","ru":"Russian","zh":"Chinese"}
sources = ['flores'] #,'wmt23'
src_root="/mnt/scratch-artemis/kshitij/LLAMA/tower-eval/TowerEval-Data-v0.1/data/raw_data/mt/"
#src_wmt23 = defaultdict(list)
src_flores = defaultdict(list)

for source in sources:
	src_path = src_root +"{}.{}".format(source,lang_pair) +"/test.jsonl"
	with jsonlines.open(src_path, "r") as r:
		for line in r.iter():
			src_flores[lang_pair].append(line["src"])

#model_path = "/mnt/data-artemis/duarte/tower_speech/tower-base-6B-5to1/llama2-7b/1500"
#model_path="/mnt/data-artemis/duarte/tower_speech/tower-base-6B-5to1/llama2-7b/3000"
#model_path = "/mnt/data-artemis/duarte/tower_speech/tower-base-8B-5to3/1500"
#model_path="/mnt/data-artemis/duarte/tower_speech/tower-base-6B-5to1/llama2-7b/1500"
#model_path="/mnt/data-artemis/duarte/tower_speech/tower-base-6B-5to1/llama2-7b/3000"
#model_path="/mnt/data-artemis/duarte/tower_speech/tower-base-6B-5to1/llama2-7b/5000"

#Tower base model
model_path="/mnt/scratch-artemis/kshitij/LLAMA/latest_megatron_codebase/tower-base-artifacts/extended_model"

model_path="/mnt/data-artemis/duarte/tower_speech/tower-base-8B-5to3/llama2-7b/3000"
tokenizer_path="/mnt/scratch-artemis/sonal/tower_speech/data/tokenizer/"
sampling_params=SamplingParams(temperature=0,stop=['\\n','\n','English:','English Text:','<\s>'], max_tokens=1024)


data_root= "/mnt/scratch-artemis/sonal/tower_speech/evaluation/instructions/5_shot/"
out_root = "/mnt/scratch-artemis/sonal/tower_speech/evaluation/results/MT/5_shot/"

#prepare into prompt list
def read_data(data_path):
    prompts_df = pd.read_json(data_path)
    ref = prompts_df["output"].to_list()
    prompts = prompts_df["instruction"].to_list()
    return prompts, ref

def vllm_generate(out_file, model_path, prompts, ref, src):
    generated  = []
    llm=LLM(model=model_path, tokenizer=tokenizer_path)
    outputs = llm.generate([prompts[i].strip() for i in range(len(prompts))], sampling_params)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        generated.append(generated_text)

        out_file.write(generated_text)
        out_file.write("\n")
        
    #compute metric
    bleu = load('bleu')
    chrf = load('chrf')
    comet = load('comet')

    results_bleu = bleu.compute(predictions=generated, references=ref)
    results_chrF = chrf.compute(predictions=generated, references=ref)
    results_comet = comet.compute(sources=src,predictions=generated, references=ref)
    results = {'bleu':results_bleu, 'chrF':results_chrF, 'comet': results_comet}
    print(results)
    return results


data_path = data_root + "{}_{}.json".format('flores',lang_pair)
out_path = out_root + "flores_base_{}".format(lang_pair)
f = open(out_path,'w')
src = src_flores[lang_pair]
prompts, ref= read_data(data_path)
results = vllm_generate(f, model_path, prompts, ref, src)
f.write("Results bleu: {} results chrF: {}".format(results['bleu'], results['chrF'], results['comet']))
