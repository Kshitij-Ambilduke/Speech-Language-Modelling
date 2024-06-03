from evaluate import load
from vllm import LLM, SamplingParams
import pandas as pd
import json
from whisper_normalizer.basic import BasicTextNormalizer
import jsonlines

normalizer = BasicTextNormalizer()
model_path = "/mnt/scratch-artemis/kshitij/LLAMA/continued_pretraining/spgi_cont_eval/only_asr_intruct_tune/checkpoint-1506"
sampling_params=SamplingParams(temperature=0,stop=['<\s>','\n'], max_tokens=1024)


data_path="/mnt/scratch-artemis/kshitij/LLAMA/continued_pretraining/Continual_pretraining__covost_data/CoVoST_complete_test.json"

out_path = "/mnt/scratch-artemis/sonal/MT-experiments/test_outputs/vllm_greedy/ASR_new.txt"


#prepare into prompt list
def read_data(data_path):
    prompts_df = pd.read_json(data_path)
    ref = prompts_df["output"].to_list()
    prompts = prompts_df["instruction"].to_list()
    return prompts, [normalizer(ref[i]) for i in range(len(ref))]

def vllm_generate(out_file, model_path, prompts, ref):
    generated  = []
    llm=LLM(model=model_path)
    outputs = llm.generate([prompts[i].strip() for i in range(len(prompts))], sampling_params)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        generated.append(normalizer(generated_text))
        
        out_file.write(generated_text)
        out_file.write("\n")
        
    #compute metric
    wer= load('wer')

    results_wer = wer.compute(predictions=generated, references=ref)
    results = {'wer':results_wer}
    print(results)
    return results




f = open(out_path,'a')
f.write("Data Path, Model Path: {} ... {}".format(data_path, model_path))

prompts, ref= read_data(data_path)
results = vllm_generate(f, model_path, prompts, ref)
f.write("Results wer: {} ".format(results['wer']))
