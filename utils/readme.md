This folder contains files used as the intermediate steps throughout the project. Following is a brief about respective files:

* `extending_model_equal2mean.py` : Extending a model such that the new embeddings (both on input and output side) are equal to the mean of the already existing embeddings. There is no randomness in the newly initialized embeddings.
* `extending_model_nonuniformly.py` : Extending a model such that the new embeddings (both input and output) are sampled from a multivariate normal distribution whose mean is equal to the mean of the already existing embeddings and has some variance.
* `km2json.py` : File used to convert kmeans data from .km files to json format which is required for training/generation. 
* `lengths.py` : File used to get the average number of DSUs in the original vs deduplicated version of the data

