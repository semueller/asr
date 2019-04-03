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
#    models = ['GRUEncoder_mfccs_100_143_tensor(0.0566)_state.tp',  'GRUEncoder_mfccs_300_29_tensor(0.0535)_state.tp',
#'GRUEncoder_mfccs_150_104_tensor(0.0567)_state.tp',  'GRUEncoder_mfccs_400_91_tensor(0.0558)_state.tp',
#'GRUEncoder_mfccs_200_56_tensor(0.0559)_state.tp','GRUEncoder_mfccs_500_28_tensor_state.tp',
#'GRUEncoder_mfccs_250_24_tensor(0.0555)_state.tp',  'GRUEncoder_mfccs_75_200_tensor(0.0623)_state.tp',
#'GRUEncoder_mfccs_25_200_tensor(0.0837)_state.tp',
    models = ['GRUEncoder_mfccs_50_200_tensor(0.0721)_state.tp']
    modelnames=[models[-1]]
#    print(modelnames); exit()
    # dataset_name = 'mfccs.pkl'

    data = pkl.load(open(datapath+dataset_name, 'rb'))
    data['X'] = torch.tensor(data['X'], dtype=torch.float)  # jibbles..
    device = check_for_gpu()
    if device.type == 'cuda':
        data['X'] = data['X'].to(device)
    # Y_cat = torch.tensor(to_categorical(data['Y']), dtype=torch.float32)

    for modelname in modelnames:

        model = load_model(modelpath, modelname, inference_only=True, dev='cpu')
#        model.to('cpu')
        print('compute encoding')
        batch_size = 256
        _, hidden_states = model.forward(data['X'][:batch_size])
        hidden_states = hidden_states.squeeze(0).to('cpu')
        for i in range(batch_size, len(data['X']), batch_size):
#            x_ = data['X'][i:i+batch_size]
            h_t = model.forward(data['X'][i:i+batch_size])[1].squeeze(0).to('cpu')
            hidden_states = torch.cat([hidden_states, h_t], dim=0)
        print(f'done {len(hidden_states)} {hidden_states[0].shape}\n\n\n')
#        h_t = torch.cat(hidden_states, dim=0).to('cpu')
#        del hidden_states
#        print(f'{h_t.shape}')
#        exit()
        print('saving')
        result = {}
        result['X'] = hidden_states
        result['Y'] = data['Y']
        result['labelranges'] = data['labelranges']
        pkl.dump(result, open(f'{datapath}/embedded/{modelname}_embedding.pkl', 'wb'))
        del result


if __name__=='__main__':
    main()
    exit(0)
