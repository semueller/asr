from asr.util import load_model, to_categorical, test_classifier, get_filenames, check_for_gpu
import click

import torch
#import numpy as np
#from sklearn.metrics.pairwise import euclidean_distances
#from umap import UMAP

#import matplotlib.pyplot as plt
import pickle as pkl
#import os


@click.command()
@click.option('--datapath', type=str, default='./path/to/data.pkl', help='expects path to pickle containing a dict with the word as key')
@click.option('--modelpath', type=str, default='./path/to/folder/with/models', help='expects path to pickle containing a dict with the word as key')
def main(modelpath, datapath, dataset_name='mfccs.pkl'):
    modelnames = get_filenames(modelpath, substr='tensor')
    # dataset_name = 'mfccs.pkl'

    data = pkl.load(open(datapath+dataset_name, 'rb'))
    x = torch.tensor(data['X'], dtype=torch.float)  # jibbles..
    device = check_for_gpu()
    if device.type == 'cuda':
        x = x.to(device)
    # Y_cat = torch.tensor(to_categorical(data['Y']), dtype=torch.float32)

    for modelname in modelnames:

        model = load_model(modelpath, modelname, inference_only=True)

        print('compute encoding')
        batch_size = 256
        hidden_states = []
        for i in range(0, len(x), batch_size):
            x_ = x[i:i+batch_size]
            _, h_t = model.forward(x)
            hidden_states.append(h_t)
        h_t = torch.stack(hidden_states, 0)

        print('saving')
        result = {}
        result['X'] = h_t
        result['Y'] = data['Y']
        result['labelranges'] = data['labelranges']
        pkl.dump(result, open(f'{datapath}/embedded/{modelname}_embedding.pkl', 'wb'))


if __name__=='__main__':
    main()
    exit(0)
