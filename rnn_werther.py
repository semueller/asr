import numpy as np
import torch
import torch.nn
import torch.nn.functional as F
import torch.optim as optim

import asr.rnn
from asr.cells import RNN, GRU
from asr.util import save_model

def load_werther(pth='./data/werther_sentences.npy'):
    werther = np.load(pth)
    return werther

def print_progress(x, progr_sym='.', progress='', scale=0.25):
    x = int(x*scale)
    p = '['+progr_sym*int(x+1)+' '*int(scale*100-x)+']'+progress
    print(p, end='\r', flush=True)

def fit(model, data):
    loss_fun = F.binary_cross_entropy
    optimizer = optim.Adam(params=model.parameters())
    epochs = 100
    data = [np.expand_dims(x, 1) for x in data]
    data = [torch.tensor(d, dtype=torch.float32) for d in data]

    for epoch in range(epochs):
        num_samples = len(data)
        for i, seq in enumerate(data):
            # feed sequence
            model.reset_state()
            optimizer.zero_grad()
            for xi, yi in zip(seq[:-1], seq[1:]):
                y_pred, new_state = model.forward(xi)
                model.update_state(new_state)
                loss = loss_fun(y_pred, yi)
                loss.backward(retain_graph=True)
            print('epoch {}/{}, seq {}/{}, loss {}'.format(epoch, epochs, i, num_samples, loss))
            optimizer.step()
        if epoch % 10 == 0:
            model.epochs_trained = i
            model.optimizer = optimizer
            save_model(model, modelname='werther_rnn')

if __name__ == '__main__':
    data = load_werther()
    in_dim = data[0][0].shape[0]
    model = RNN(in_dim, hidden_dim=250, out_dim=in_dim, activation_function_out=torch.sigmoid)
    fit(model, data)
    save_model(model, modelname='werther_rnn')
