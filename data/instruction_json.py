import json
import jsonlines
ref_path="/mnt/scratch-artemis/kshitij/LLAMA/tower-eval/TowerEval-Data-v0.1/data/raw_data/mt/flores.en-de/test.jsonl"
instruction_path= "/mnt/scratch-artemis/kshitij/LLAMA/tower-eval/TowerEval-Data-v0.1/data/instructions/5_shot/mt/flores.en-de/instructions.txt"

outfile= "instruction_five_shot_flores.json"
output = []

refs = []
with jsonlines.open(ref_path, "r") as r:
	#data = json.load(r)
	for line in r.iter():
		#data = json.load(line)
		refs.append(line["ref"])
with open(instruction_path, "r") as f:
	for i,line in enumerate(f.readlines()):
		outdict = {"instruction":line, "output":refs[i]}
		output.append(outdict)
with open(outfile,'w+') as f:
    json.dump(output, f, indent=6, ensure_ascii=False)
