The files in this folder can be used to train a tokenizer on the DSUs. For doing this following strategy was adopted:

1. For representing each DSU as a single symbol, they were deteministically mapped to characters in private unicode space. These private unicode spaces range from:
    * U+E000 to U+F8FF 
    * U+F0000 to U+FFFFD
    * U+100000 to U+10FFFD
2. By doing this, you can train the tokenizer model on these new mappings and after decoding, if you want the DSU back, we can map the unicode characters back to DSUs.
3. After training this tokenizer, the original tokenizer model from huggingface has to be combined with this new tokenizer model to get a tokenizer which is able to tokenize both text and speech.
4. Finally, this sentencepiece model has to be converted into huggingface tokenizer.

Steps 1 and 2 along with training the tokenizer is done by `sentencepiece_trainer.py`. Step 3 of merging the original and new vocabs is done by `merge_vocabs.py`. Step 4 of converting the sentencepiece model to huggingface is trivial and can be done using `sentencepiece2hf_tokenizer.py`.

For utilizing this new tokenizer, the original data from the DSUs has to be converted into their corresponding mappings into the private unicode space. This can be done using `formatting_data.py`. This file gives the final json file while can be used for training.

The training of the tokenizer can be done either on original DSU data or on deduplicated version of the original data. For the later, the data used for training the tokenizer can be made using `deduplicating_kmeans.py`