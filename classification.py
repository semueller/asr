import sys
import click
import pickle as pkl

from asr.util import save_model
from asr.encoder import GRUEncoder, LSTMEncoder

import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam


def to_categorical(Y):
    class_labels = np.unique(Y)
    num_classes = len(class_labels)
    e = np.eye(num_classes)
    codebook = {l: i for i, l in enumerate(class_labels)}
    categories = np.array([e[codebook[l]] for l in Y])
    return categories


@click.command()
@click.option('--data', type=str, default='./path/to/data.pkl', help='expects path to pickle containing a dict with the word as key')
@click.option('--model', type=str, default='./save/model/to/', help='expects path to pickle containing a dict with the word as key')
def main(data, model):

    print('loading data from {}'.format(data))
    filename = data.split('/')[-1].split('.')[0]
    data = pkl.load(open(data, 'rb'))
    print(data.keys())


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Additional Info when using cuda
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_cached(0) / 1024 ** 3, 1), 'GB')

    train_percentage = 0.8
    x = data['X']
    Y = data['Y']
    Y = to_categorical(Y)
    seqlen, nfeatures = x.shape[1:]
    idxs_train = [i for i in range(len(x))]

    # np.random.shuffle(idxs)
    # split = np.math.ceil(len(idxs)*train_percentage)
    # idxs_train, idxs_test =  idxs[:split], idxs[split:]
    # with open(filename+'_idxs.dat', 'w') as f:
    #     f.write('train: \n {} \n test: \n {} \n'.format(idxs_train, idxs_test))

    x_train = torch.tensor(x[idxs_train], dtype=torch.float)
    y_train = torch.tensor(Y[idxs_train], dtype=torch.float)
    n_samples, num_classes = y_train.shape
    del x


    hidden_size = 500
    n_epochs = 1000
    network = GRUEncoder(input_size=nfeatures, hidden_size=hidden_size, out_dim=num_classes,
                         act_out=nn.Sigmoid)

    if device.type == 'cuda':
        x_train = x_train.to(device)
        y_train = x_train.to(device)
        network = network.to(device)

    optim = Adam(network.parameters())
    loss_fun = nn.BCELoss()
    histories = []
    target_loss = 1e-2
    train = True

    batch_size = 256
    epochs = 0
    while train:

        history = []

        for i in range(0, n_samples, batch_size):
            x, y = x_train[i: i+batch_size], y_train[i: i+batch_size]

            optim.zero_grad()

            y_pred, _ = network.forward(x)
            loss = loss_fun(y_pred, y)

            if i/batch_size % 10 == 0:
                print(f'epoch {epochs} {i}:{i+batch_size}/{n_samples} loss {loss}')

            loss.backward()
            history.append(loss)
            optim.step()

        epochs += 1

        current_performance = torch.mean(torch.tensor(history[-50:-1]))
        train = target_loss > current_performance
        histories.append(histories)




    modelname = '_'.join([network.__class__.__name__, filename, str(hidden_size), str(n_epochs)])
    save_model(network, path=model, modelname=modelname)
    pass

if __name__ == '__main__':
    main()
    sys.exit(0)
