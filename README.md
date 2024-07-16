## Tower Speech: Continued Pretraining of DSUs

Kshitij Added:
1. Instruction tuning codebase
2. Tokenzier training code
3. Some utils

To add by Kshitij:
1. kmeans data creation pipeline from fairseq 
=======

### Evaluation

Evaluation is done on ASR and MT where we report WER and BLEU/ChrF/Comet respectively for each task. [ToDo: Speech Translation]
./scripts/ includes slurm scripts as well as vLLM generation scripts for ASR and MT

$ python3 ./scritps/vllm_ASR.py --token_path "" --model_path "" --data_path ""

 will produce the output file in location (edit in vllm_ASR.py file)
[ToDo: arg parse for MT]

Some outputs from models tested with can also be found in ./outputs/

Outputs from VLLM are in ./outputs/vllm_greedy with appropriate information regarding data used and model used on the head of the file, and results in the tail (Comet, ChrF, BLEU)

Laslty, the instructions used can be found in ./data/
