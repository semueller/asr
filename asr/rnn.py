import os
import sys
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from .cells import RNN, GRU, LSTM

# Recurrent Network class

class RecurrentNetwork(nn.Module):
    def __init__(self, celltype, in_dim, hidden_dim, out_dim, layers=1):
        super(RecurrentNetwork, self).__init__()
        celltype = celltype.lower()
        self.cell = None
        if celltype == 'rnn':
            self.cell = RNN(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim)
        elif celltype == 'gru':
            self.cell = GRU(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim)
        elif celltype == 'lstm':
            self.cell = LSTM()
        else:
            raise ValueError('Celltype {} not recognized'.format(celltype))

    def forward(self, x):
        y, new_state = self.cell.forward(x)
        self.cell.a = new_state
        return y, new_state

    def fit(self, x, labels):
        pass