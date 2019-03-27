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

def error_rate(pred, target):
    m = torch.argmax(pred, 1)
    z = torch.zeros(pred.shape)
    for r, i in zip(z, m):
        r[i] = 1
    t = torch.mean(torch.abs(z-target), 1) != 0
    return torch.mean(t)

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

    train_percentage = 0.95
    x = data['X']
    Y = data['Y']
    Y = to_categorical(Y)
    seqlen, nfeatures = x.shape[1:]
    idxs = [i for i in range(len(x))]

    np.random.shuffle(idxs)
    split = np.math.ceil(len(idxs)*train_percentage)
    idxs_train, idxs_test =  idxs[:split], idxs[split:]
    with open(filename+'_idxs.dat', 'w') as f:
        f.write('train: \n {} \n test: \n {} \n'.format(idxs_train, idxs_test))

    x_train = torch.tensor(x[idxs_train], dtype=torch.float)
    y_train = torch.tensor(Y[idxs_train], dtype=torch.float)
    x_test = torch.tensor(x[idxs_test], dtype=torch.float)
    y_test = torch.tensor(Y[idxs_test], dtype=torch.float)

    n_samples, num_classes = y_train.shape
    del x


    hidden_size = 500
    n_epochs = 1000
    network = GRUEncoder(input_size=nfeatures, hidden_size=hidden_size, out_dim=num_classes,
                         act_out=nn.Sigmoid)

    if device.type == 'cuda':
        x_train = x_train.to(device)
        y_train = y_train.to(device)
        y_test = y_test.to(device)
        x_test = x_test.to(device)
        network = network.to(device)

    optim = Adam(network.parameters())
    loss_fun = nn.BCELoss()
    histories = []
    target_performance = 1e-2
    train = True

    batch_size = 0
    n_epochs = 0

    while train:

        history = []
        if batch_size > 0:
            for i in range(0, n_samples, batch_size):
                x, y = x_train[i: i+batch_size], y_train[i: i+batch_size]

                optim.zero_grad()

                y_pred, _ = network.forward(x)
                loss = loss_fun(y_pred, y)

                if i/batch_size % 10 == 0:
                    print(f'epoch {n_epochs} {i}:{i+batch_size}/{n_samples} loss {loss}', end='\r', flush=True)

                loss.backward()
                history.append(loss)
                optim.step()
            y_pred_test, _ = network.forward(x_test)
            current_performance = error_rate(y_pred_test, y_test)
            print(f'\ncurrent performance on test set{current_performance}')
        else:
            optim.zero_grad()

            y_pred, _ = network.forward(x_train)
            loss = loss_fun(y_pred, y_train)

            if n_epochs % 10 == 0:
                print(f'epoch {n_epochs} loss {loss}', end='\r', flush=True)

            loss.backward()
            history.append(loss)
            optim.step()

        n_epochs += 1

        train = target_performance < current_performance
        histories.append(histories)



    modelname = '_'.join([network.__class__.__name__, filename, str(hidden_size), str(n_epochs)])
    save_model(network, path=model, modelname=modelname)
    pkl.dump(histories, open(f'{model}_{modelname}_history.pkl', 'wb'))


if __name__ == '__main__':
    main()
    sys.exit(0)
