## Installation and Experiments

### I. Installation
* set up a python environment (eg with conda) with python 3.5+
* install the project with pip (or manually install requirements listed in setup.py):
```sh
> pip install -e ./
```
<br>

### II. Data
##### Download
Downloads the Tensorflow Speechcommands dataset and extracts it to './data/tf'
```sh
> ./download_data.sh
```
##### Preprocessing
You need to merge all classes (excluding _background_noise_) into the Dataset classe defined in asr.util.
You can merge all datapoints into a zip file containing each word in its own archive with the words as keys and use prep_data.py to load it and call zip_to_mfcc() on it.
<br>


### III. Experiments
You need to run those scripts:
* classification.py: trains rnn classifiers for mfccs
* dataset_to_hidden_states.py: runs data through trained classifier and saves the hidden states for each datapoint (-> discrete, 'embeddings')
* train_gmlvq_embedding.py: train gmlvq classifier on the hidden states
* evaluation.py:
** modeltype = classifier: compute and plot distance matrices on embeddings
** modeltype = gmlvq: compute classifier accuracy on testset and visualize relevance profile and relevance matrix

```sh
> python classification.py --datapath /path/to/mffcs.pkl --modelpath /destination/to/save/to

> python dataset_to_hidden_states.py --datapath /path/to/folder --dataset_name filname --modelpath /path/to/nn_classifiers

> python train_gmlvq_embedding --datapath /path/to/embeddings --modelpath /destination/to/save/to --idxs /path/to/file_test_idxs.pkl

> python evaluation.py --modeltype classifier --datapath /path/to/mfccs_subset.pkl --modelpath /path/to/classifiers
> python evaluation.py --modeltype gmlvq --datapath /path/to/embeddings --modelpath /path/to/gmlvq_models
```

The scripts produce results for embeddings with 25, 50 and 250 dimensions.
I took the 500 dimension embedding out because of memory issues when converting the dataset to its embedding as well as time concerns becuase the gmlvq model's distance matrix is quadratic wrt the data dimensionality.
