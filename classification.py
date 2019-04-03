import os
import sys
import click
import pickle as pkl

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from asr.util import save_model, test_classifier, Dataset, load_pkl
from asr.encoder import GRUEncoder, LSTMEncoder


@click.command()
@click.option('--datapath', type=str, default='./path/to/data.pkl', help='expects path to pickle containing a dict with the word as key')
@click.option('--modelpath', type=str, default='./save/model/to/', help='expects path to pickle containing a dict with the word as key')
def main(datapath, modelpath):

    print(f'loading data from {datapath}')
    filename = datapath.split('/')[-1].split('.')[0]
#    print(datapath.split('/')[:-1])
#    exit()
    datadir = datapath.split('/')[:-1]
    datadir = os.path.join(*datadir)
    dataset = load_pkl(datapath)  # Dataset from asr.util
    print('looking for idx files')
    idxs_train = load_pkl(os.path.join('/', datadir, 'idxs_train.pkl'))
    idxs_test = load_pkl(os.path.join('/', datadir, 'idxs_test.pkl'))


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Additional Info when using cuda
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_cached(0) / 1024 ** 3, 1), 'GB')

    x = dataset.data
    Y = dataset.get_labels_categorical()
    seqlen, nfeatures = x.shape[1:]
    print('splitting up data into train and test sets')
    x_train = torch.tensor(x[idxs_train], dtype=torch.float)
    y_train = torch.tensor(Y[idxs_train], dtype=torch.float)
    x_test = torch.tensor(x[idxs_test], dtype=torch.float)
    y_test = torch.tensor(Y[idxs_test], dtype=torch.float)
    del x, Y

    n_samples, num_classes = y_train.shape

    x_train = x_train.to(device)
    y_train = y_train.to(device)
    y_test = y_test.to(device)
    x_test = x_test.to(device)


    # hidden_size = [25, 50, 75, 100, 150, 200, 250, 300, 400]
    hidden_size = [500, 250, 50, 25]
    print(f'starting training of classifiers with hidden dims of {hidden_size}')

    for hid_s in hidden_size:
        print(f'training model with {hid_s} hidden dims')
        network = GRUEncoder(input_size=nfeatures, hidden_size=hid_s, out_dim=num_classes,
                             act_out=nn.Sigmoid, num_layers=1)

        if device.type == 'cuda':
            network = network.to(device)

        optim = Adam(network.parameters())
        loss_fun = nn.BCELoss()
        histories = []
        target_error = 57e-3
        train = True

        batch_size = 256
        n_epochs, max_epochs = 0, 200
        print(f'test performances without training: {test_classifier(network.forward, x_test, y_test, batch_size)}')

        while train:
            history = []

            for i in range(0, n_samples, batch_size):
                x, y = x_train[i: i+batch_size], y_train[i: i+batch_size]
                optim.zero_grad()
                y_pred, _ = network.forward(x)
                loss = loss_fun(y_pred, y)
                loss.backward()
                history.append(loss)
                optim.step()
                if i/batch_size % 10 == 0:
                    print(f'epoch {n_epochs} {i}:{i+batch_size}/{n_samples} loss {loss}', end='\r', flush=True)



            current_error = test_classifier(network.forward, x_test, y_test, batch_size)
            print(f'\ntest error: {current_error} \n')

            n_epochs += 1

            train = target_error < current_error and n_epochs < max_epochs
            histories.append(history)

        modelname = '_'.join([network.__class__.__name__, filename, str(hid_s), str(n_epochs), str(current_error)])
        network.optimizer = optim
        network.history = histories
        network.epochs_trained = n_epochs
        save_model(network, path=modelpath, modelname=modelname)


if __name__ == '__main__':
    main()
    sys.exit(0)
