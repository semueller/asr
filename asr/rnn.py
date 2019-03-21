import os
import sys
import numpy as np

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

from .cells import RNN, GRU, LSTM, Cell

# Recurrent Network class

class RecurrentNetwork(nn.Module):
    def __init__(self, celltype, in_dim, hidden_dim, out_dim, layers=1):
        super(RecurrentNetwork, self).__init__()
        celltype = celltype.lower()
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.cell = None
        if celltype == 'rnn':
            self.cell = RNN(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim)
        elif celltype == 'gru':
            self.cell = GRU(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim)
        elif celltype == 'lstm':
            self.cell = LSTM(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim)
        else:
            raise ValueError('Celltype {} not recognized'.format(celltype))

        self.loss_function = nn.MSELoss()
        self.optimizer = optim.Adam(params=self.parameters(), lr=10)

    def forward(self, x):
        y, new_state = self.cell.forward(x)
        self.cell.a = new_state
        return y, new_state

    def fit(self, X, Y, n_epochs=10, threshold=1E-2):

        if len(X) != len(Y):
            raise ValueError('number of time series and number of labels did not match!')

        histories = []
        for epoch in range(n_epochs):

            history = []
            for i, (x, y) in enumerate(zip(X, Y), 0):

                self.cell.init_state()
                self.optimizer.zero_grad()
                y_pred = []
                for x_i in x:
                    y_, _ = self.forward(x_i)
                    y_pred.append(y_)
                y_pred = torch.tensor(y_pred, dtype=y.dtype).reshape(y.shape)
                loss = self.loss_function(y_pred, y)
                history.append(loss.item())
                if i % 50 == 0:
                    print('epoch {}, loss after {} series: {}'.format(epoch, i+(epoch*len(X)), history[-1]))
                    # print(self.cell._modules['w_in'].weight.data)
                    # print(self.cell.a)
                loss.backward()
                self.optimizer.step()

            histories.append(history)
            if np.mean(history) <= threshold:
                break
        return histories