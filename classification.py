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
    _p = torch.argmax(pred, 1)
    _t = torch.argmax(target, 1)
    error = 1-torch.mean((_p == _t).to(torch.float32))
    return error

def test_model(model, x, y, batch_size=256):
    with torch.no_grad():
        errors = []
        for i in range(0, len(x), batch_size):
            x_, y_ = x[i:i+batch_size], y[i:i+batch_size]
            y_p, _ = model.forward(x_)
            errors.append(error_rate(y_p, y_))
    return torch.mean(torch.tensor(errors))

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
    del x, Y

    n_samples, num_classes = y_train.shape


#    n_epochs = 1000

    hidden_size = [25, 50, 75, 100, 150, 200, 250, 300, 400]
    network = GRUEncoder(input_size=nfeatures, hidden_size=hidden_size, out_dim=num_classes,
                         act_out=nn.Sigmoid, num_layers=1)

    if device.type == 'cuda':
        x_train = x_train.to(device)
        y_train = y_train.to(device)
        y_test = y_test.to(device)
        x_test = x_test.to(device)
        network = network.to(device)

    optim = Adam(network.parameters())
    loss_fun = nn.BCELoss()
    histories = []
    target_error = 57e-3
    train = True

    batch_size = 256
    n_epochs, max_epochs = 0, 100
    print(f'test performances without training: {test_model(network, x_test, y_test, batch_size)}')

    while train:

        history = []

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

        current_error = test_model(network, x_test, y_test, batch_size)
        print(f'\ntest error: {current_error} \n')

        n_epochs += 1

        train = target_error < current_error and n_epochs < max_epochs
        histories.append(history)



    modelname = '_'.join([network.__class__.__name__, filename, str(hidden_size), str(n_epochs), str(curent_error)])
    network.optimizer = optim
    network.epochs_trained = n_epochs
    network.history = histories
    save_model(network, path=model, modelname=modelname)
    pkl.dump(histories, open(f'{model}_{modelname}_history.pkl', 'wb'))


if __name__ == '__main__':
    main()
    sys.exit(0)
