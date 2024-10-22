from evaluate import load
from vllm import LLM, SamplingParams
import pandas as pd
import json
import jsonlines

src_path="/mnt/scratch-artemis/kshitij/LLAMA/tower-eval/TowerEval-Data-v0.1/data/raw_data/mt/wmt23.en-de/test.jsonl"

src_wmt23 = []
with jsonlines.open(src_path, "r") as r:
	for line in r.iter():
		src_wmt23.append(line["src"])


src_path="/mnt/scratch-artemis/kshitij/LLAMA/tower-eval/TowerEval-Data-v0.1/data/raw_data/mt/flores.en-de/test.jsonl"

src_flores = []
with jsonlines.open(src_path, "r") as r:
        for line in r.iter():
                src_flores.append(line["src"])
model_paths = {"text_ext":'/mnt/scratch-artemis/kshitij/LLAMA/Megatron_LLM/only_text_experiment_with_extension/Hf_ckpt/tinyllama-1b/1000',"cpt": '/mnt/scratch-artemis/kshitij/LLAMA/Megatron_LLM/NONUNIFORM_1B_EXPERIMENT/Hf_ckpt/tinyllama-1b/4660',"llama-7b":'/mnt/data-shared/models/Llama-2-7b-hf', "tiny-1b":'/mnt/scratch-artemis/sonal/MT-experiments/TinyLlama-1.1B-intermediate-step-1431k-3T', "text":'/mnt/scratch-artemis/kshitij/LLAMA/Megatron_LLM/only_text_experiment/Hf_ckpt/tinyllama-1b/1000'}

#model_paths = {"text":model_paths["text"]}
#model_paths = {"llama": model_paths["llama-7b"]}
sampling_params=SamplingParams(temperature=0,stop=['\\n','\n','English:','English Text:','<\s>'], max_tokens=1024)
sources = {"wmt_0":src_wmt23, "wmt_5":src_wmt23, "flores_0":src_flores, "flores_5":src_flores}
data_root= "/mnt/scratch-artemis/sonal/MT-experiments/"

data={"flores_5":'instruction_five_shot_flores.json',"flores_0":'instruction_zero_shot.json',"wmt_5":'instruction_5shot.json',"wmt_0":'instruction_zero_shot_wmt.json'}
out_root = "/mnt/scratch-artemis/sonal/MT-experiments/test_outputs/vllm_greedy/"


#prepare into prompt list
def read_data(data_path):
    prompts_df = pd.read_json(data_path)
    ref = prompts_df["output"].to_list()
    prompts = prompts_df["instruction"].to_list()
    return prompts, ref

def vllm_generate(out_file, model_path, prompts, ref, src):
    generated  = []
    llm=LLM(model=model_path)
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
    results = {'bleu':results_bleu, 'chrF':results_chrF , 'comet':results_comet}
    print(results)
    return results

for model_name,model_path in model_paths.items():
    for data_name,data_path in data.items():
        data_path = data_root + data_path
        out_path = out_root + "vllm_{}_{}".format(model_name, data_name)
        f = open(out_path,'a')
        f.write("Data Path, Model Path: {} ... {}".format(data_path, model_path))
        src = sources[data_name]
        prompts, ref= read_data(data_path)
        results = vllm_generate(f, model_path, prompts, ref, src)

        f.write("Results bleu: {} results chrF: {} results comet: {} ".format(results['bleu'], results['chrF'], results['comet']))
        
