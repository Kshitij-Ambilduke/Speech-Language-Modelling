from itertools import groupby
import json
import argparse

def main(args):
    with open(args.kmeans_path) as f:
        kmeans_data = f.readlines()
    kmeans_data = [i.strip() for i in kmeans_data]

    if args.deduplicate_before:
        for i in range(len(kmeans_data)):
            point = kmeans_data[i]
            point = point.split()
            point = [int(j) for j in point]
            point = [x for x,_ in groupby(point)]
            kmeans_data[i] = " ".join(point)

    with open(args.transcript_path) as f:
        transcript_data = f.readlines()
    transcript_data = [i.strip() for i in transcript_data]

    for i in range(len(kmeans_data)):
        kmeans_data[i] = [int(j) for j in kmeans_data[i].split()]
        kmeans_data[i] = [chr(t + args.private_key) for t in kmeans_data[i]]    #chr(): input: unicode, output: character
        kmeans_data[i] = "".join(kmeans_data[i])
        kmeans_data[i] = "English speech: " + kmeans_data[i] + " English text:"

    assert len(kmeans_data)==len(transcript_data)
    print(f"Length of dataset = {len(kmeans_data)}")

    final_data=[]
    for n,i in enumerate(kmeans_data):
        final_data.append(
            {
                "instruction":i,
                "output":transcript_data[n]
            }
        )

    with open(args.output_path,'w') as f:
        json.dump(final_data,f,ensure_ascii=False, indent=6)

if __name__=="__main__":
    '''
    args.private_key = 990000
    args.kmeans_path = "/home/kshitij/LLAMA/continued_pretraining/covost_data/2000/deduplicated_data/train_0_1.km"
    args.transcript_path= "/home/kshitij/LLAMA/continued_pretraining/covost_data/train_transcripts.txt"
    args.output_path = "/home/kshitij/LLAMA/continued_pretraining/bpe_tokenizer/formatted_data_2kdedup_to_8k/train.json"
    args.deduplicate_before = True
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--private-key")
    parser.add_argument("--kmeans-path")
    parser.add_argument("--transcript-path")
    parser.add_argument("--output-path")
    parser.add_argument("--deduplicate-before", type=bool)
    args = parser.parse_args()
    main(args)