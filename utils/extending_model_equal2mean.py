import transformers
import torch
import argparse

# args.new_vocab_size = 37005
# args.original_model_path = "/home/kshitij/.cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-intermediate-step-1431k-3T/snapshots/036fa4651240b9a1487f709833b9e4b96b4c1574/"
# args.output_model_path = "/mnt/scratch-artemis/kshitij/LLAMA/Megatron_LLM/temp/extended_model_new_output_zeros"

def main(args):
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.original_model_path)
    model = transformers.AutoModelForCausalLM.from_pretrained(args.original_model_path)
    number_new_tokens = args.new_vocab_size-len(tokenizer)
    model.resize_token_embeddings(args.new_vocab_size)

    if number_new_tokens > 0:
        print("smartinit")
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-number_new_tokens].mean(
            dim=0, keepdim=True
        )
        output_embeddings_avg = output_embeddings[:-number_new_tokens].mean(
            dim=0, keepdim=True
        )

        input_embeddings[-number_new_tokens:] = input_embeddings_avg
        output_embeddings[-number_new_tokens:] = output_embeddings_avg

        #For setting the new output_embeddings corresponding to sound tokens 
        # output_embeddings[-number_new_tokens:] = torch.zeros(output_embeddings_avg.shape)
        
    model.save_pretrained(args.output_model_path)
    # check = model.get_input_embeddings().weight.data
    # print(check[-number_new_tokens:])
    # print(model.lm_head.weight)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-vocab-size", help="Increased vocab size")
    parser.add_argument("--original-model-path", help="Original model location/name")
    parser.add_argument("--output-model-path", help="path to save extended model")
    opt = parser.parse_args()
    main(opt)