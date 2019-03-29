import pickle as pkl
from asr.util import load_model, to_categorical, test_classifier, get_filenames
import click

import torch
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from umap import UMAP

import matplotlib.pyplot as plt
import pickle as pkl
import os


@click.command()
@click.option('--data', type=str, default='./path/to/data.pkl', help='expects path to pickle containing a dict with the word as key')
@click.option('--models', type=str, default='./path/to/folder/with/models', help='expects path to pickle containing a dict with the word as key')
def main(modelpath, datapath, dataset_name='mfccs.pkl'):
    modelnames = get_filenames(modelpath, substr='tensor')
    # dataset_name = 'mfccs.pkl'

    data = pkl.load(open(datapath+dataset_name, 'rb'))
    data['X'] = torch.tensor(data['X'], dtype=torch.float)  # jibbles..
    # Y_cat = torch.tensor(to_categorical(data['Y']), dtype=torch.float32)

    for modelname in modelnames:

        print(f'\n\n\nevaluating {modelname}')
        model = load_model(modelpath, modelname)

        print('compute encoding')
        _, h_t = model.forward(data['X'])

        result = {}
        result['X'] = h_t
        result['Y'] = data['Y']
        result['labelranges'] = data['labelranges']
        pkl.dump(result, open(f'{datapath}/embedded/{modelname}_embedding.pkl', 'wb'))


if __name__=='__main__':
    main()
    exit(0)