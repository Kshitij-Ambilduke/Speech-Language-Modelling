# Tower Speech: Continued Pretraining of DSUs

This repo contains training and evaluation scripts to reproduce TowerSpeech, a model continually pretrained over Tower (Alves et al., 2024) on Discrete Speech Units (DSUs) and text data across 10 languages (En, Es, Fr, It, Pt, Ru, Zh, De, Nl, Ko). 

## Data Mixture 

We use a combination of resources to create our data mixture of varing proportions. Primarily:

* English Speech: Multilingual Libri Speech, GigaSpeech, SPGISpeech, VoxPopuli
* Text: Multilingual and monolingual data subsampled from Tower 

We report various mixtures as follows:

*TowerSpeech 6B: We use Tower 7B as our base model and use 5B tokens of speech data and 1B tokens of text data. 
*LlamaSpeech 6B: We use Llama-2 7B as our base model with the same data mixture. 

Preliminary bilingual (En, De) models on a smaller scale include:

*TinyLlama 10B: We use TinyLlama 1B as our base model with 5B tokens of speech data and 5B tokens of monolingual and parallel text in English and German. We utilise a subsplit of Tower here as well.
*Llama-2 10B: We use Llama-2 as our base model with 5B tokens of speech data and 5B tokens of text.

## Training 

This repository contains code to extend the tokeniser of the model and train one on DSUs (train_bpe_tokenizer). Following this, code to train the model can be found here: https://github.com/deep-spin/multilinguality_megatron/tree/tower-speech 

Once a model is trained, we apply instruction tuning for which the code is reported here as well (instruction_tuning). 

## Evaluation

The instructions used for evaluation can be found pre-formatted at /instructions where 0_shot and 5_shot are subdirectories that contain 0 and 5 shot instructions for machine translation using Flores and WMT23. 

The folder /generate contains vllm generation scripts used for MT and ASR generation (generate.py for MT and generate_ASR.py for ASR). 

The relevant sbatch scripts show how you can run these scripts in command line.

The folder /results reports results at all checkpoints for different data mixes (as in the name, 5to1 refers to a 6b data mix with 5b speech and 1b text tokens similarly 5to3 indicates 5b tokens of speech and 3b tokens of text.)
