import transformers


sentencepiece_tokenzier_path = (
    "/home/kshitij/LLAMA/continued_pretraining/bpe_tokenizer/llama_plus_sound/llama2_2kdedup_to_8k.model"
)

tokenizer = transformers.LlamaTokenizer(vocab_file=sentencepiece_tokenzier_path, legacy=False)
print(len(tokenizer))
tokenizer.save_pretrained(
    "/home/kshitij/LLAMA/continued_pretraining/bpe_tokenizer/trained_tokenizers/_BPE_2kdedup_to_8k_HF"
)
