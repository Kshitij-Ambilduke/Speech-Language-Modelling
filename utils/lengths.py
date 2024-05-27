import sys
from itertools import groupby
from statistics import mean


def main(k_means_path):
    with open(k_means_path) as f:
        data = f.readlines()

    data = [i.strip().split() for i in data]

    original_lengths = []
    for i in data:
        original_lengths.append(len(i))
    original_lengths_mean = mean(original_lengths)

    for j in range(len(data)):
        data[j] = [int(i) for i in data[j]]
        data[j] = [x for x, _ in groupby(data[j])]

    deduplicated_lengths = []
    for i in data:
        deduplicated_lengths.append(len(i))
    deduplicated_lengths_mean = mean(deduplicated_lengths)

    print(f"Original length: {original_lengths_mean}")
    print(f"Deduplicated length: {deduplicated_lengths_mean}")


if __name__ == "__main__":
    k_means_path = sys.argv[1]
    # k_means_path = "/mnt/scratch/kshitij/GigaSpeech/kmeans_lab_dir/split_15_0_1.km"
    # PATH TO THE .km FILE FOR FINDING AVG LENGTH OF ORIGINAL AND DEDUPLICATED DSUs
    main(k_means_path)
