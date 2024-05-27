import os
import sys

import sentencepiece as spm


# print(sys.argv)
vocab_size, model_type, model_prefix, train_data_path, output_dir = sys.argv[1:]
vocab_size = int(vocab_size)
model_prefix = os.path.join(output_dir, model_prefix)

# train_data_path = path to the kmeans file "/home/kshitij/LLAMA/data/gitlab/multimodal-llm-data/Librispeech/hubert-base-9th-2000/train_100h_0_1.km"

"""
 Mapping the training data to private unicode space.
 > Offsetting these input cluster IDs to private unicode space using a private key. Private ranges are:
 > U+E000 – U+F8FF  
 > U+F0000 – U+FFFFD
 > U+100000 – U+10FFFD
 > Then these unicode IDS are converted to string to get private unicode characters
"""

private_offset = 990000

# Mapping the cluster ids to unicode chars and offsetting to private space
with open(train_data_path, "r") as f:
    data = f.readlines()

for i in range(len(data)):
    data[i] = [int(j) for j in data[i].strip().split()]
    data[i] = [chr(t + private_offset) for t in data[i]]  # chr(): input: unicode, output: character
    data[i] = "".join(data[i]) + "\n"

with open(os.path.join(output_dir, "unicode_corpus.txt"), "w+") as f:
    f.writelines(data)

# Training the tokenizer
spm.SentencePieceTrainer.train(
    input=os.path.join(output_dir, "unicode_corpus.txt"),
    model_prefix=model_prefix,
    model_type=model_type,
    vocab_size=vocab_size,
    character_coverage=1.0,
    eos_id=-1,
    add_dummy_prefix=False,
    unk_id=2,
    bos_id=-1,
    max_sentencepiece_length=256,
    pad_id=3,
)

## For using this tokenizer: Make sure that you map the input to the same private space first and then apply the tokenizer on that.
