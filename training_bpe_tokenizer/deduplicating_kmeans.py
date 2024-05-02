import argparse
from itertools import groupby


# args.original_kmeans_path = "/home/kshitij/LLAMA/continued_pretraining/covost_data/2000/test_0_1.km"
# args.deduplicated_kmeans_path = "/home/kshitij/LLAMA/continued_pretraining/covost_data/2000/deduplicated_data/test_0_1.km"


def main(args):
    with open(args.original_kmeans_path) as f:
        original_data = f.readlines()

    original_data = [i.strip() for i in original_data]
    print(len(original_data))
    print()

    for i in range(len(original_data)):
        point = original_data[i]
        point = point.split()
        point = [int(j) for j in point]
        point = [str(x) for x, _ in groupby(point)]
        point = " ".join(point)
        original_data[i] = point

    print(len(original_data))

    with open(args.deduplicated_kmeans_path, "w") as f:
        for i in original_data:
            f.write(i + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-kmeans-path")
    parser.add_argument("--deduplicated-kmeans-path")
    args = parser.parse_args()
    main(args)
