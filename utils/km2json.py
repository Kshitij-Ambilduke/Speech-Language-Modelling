import argparse
import json
from itertools import groupby


def main(args):

    with open(args.kmeans_path, "r") as f:
        kmeans = f.readlines()
    kmeans = [i.strip() for i in kmeans]

    print(len(kmeans))

    with open(args.transcript_path, "r") as f:
        json_data = f.readlines()

    json_data = [json_data[i].strip() for i in range(len(json_data))]
    print(len(json_data))

    # FOR args.DEDUPLICATION
    if args.deduplication:
        for i in range(len(kmeans)):
            point = kmeans[i]
            point = point.split()
            point = [int(j) for j in point]
            point = [x for x, _ in groupby(point)]
            point = "".join([f"<extra_id_{j}>" for j in point])
            kmeans[i] = point
    else:
        for i in range(len(kmeans)):
            point = kmeans[i]
            point = point.split()
            point = " ".join([f"<|<SOUND_{j}>|>" for j in point])
            kmeans[i] = point

    assert len(json_data) == len(kmeans)

    op_data = []
    for i in range(len(kmeans)):
        op_data.append({"instruction": f"English speech: {kmeans[i]}. English text:", "output": json_data[i]})

    with open(args.output_path, "w+") as f:
        json.dump(op_data, f, indent=6, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kmeans-path")
    parser.add_argument("--output-path")
    parser.add_argument("--transcript-path")
    parser.add_argument("--args.deduplication", type=bool, default=True)
    args = parser.parse_args()
    main(args)
