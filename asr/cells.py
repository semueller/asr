import os
import sys
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

class Cell(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, activation_function=F.tanh,
                 activation_function_out=nn.relu, state_initializer=None):
        super(Cell, self).__init__()
        self.in_dim = in_dim
        self.state_size = hidden_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.activation_function = activation_function
        self. act_out_fun = activation_function_out
        self.state_init = state_initializer
        self.a = None
        self.init_state()  # state

    def update_state(self, s):
        self.a = s

    def init_state(self, initializer=None):
        if self.state_init is not None:
            if callable(self.state_init):
                self.a = self.state_init(self.hidden_dim)
                assert self.hidden_dim in self.a.shape
            if isinstance(self.state_init, int):
                self.a = torch.zeros(self.hidden_dim) + initializer
        else:
            self.a = torch.zeros(self.hidden_dim)



# Simple Recurrent Cell
class RNN(Cell):
    def __init__(self, in_dim, hidden_dim, out_dim, activation_function=F.tanh,
                 activation_function_out=nn.relu, state_initializer=None):
        # in_dim = number of featurse for one element of time series
        super(RNN, self).__init__(in_dim, hidden_dim, out_dim, activation_function=activation_function,
                 activation_function_out=activation_function_out, state_initializer=state_initializer)
        self.w_in = nn.Linear(in_features=self.state_size+in_dim,
                              out_features=self.state_size+in_dim)
        self.w_y = nn.Linear(self.state_size, out_dim)

    def forward(self, x, state):
        ax = self.w_in(torch.cat((state, x)))
        new_state = self.act_fun(ax)
        y = self.w_out(new_state)  # ? already use the new a here?
        y = self.act_fun_out(y)
        return y, new_state

# GRU cell ???
class GRU(Cell):
    def __init__(self, in_dim, hidden_dim, out_dim, activation_function=F.tanh,
                 activation_function_out=nn.relu, state_initializer=None):
        # in_dim = number of featurse for one element of time series
        super(GRU, self).__init__(in_dim, hidden_dim, out_dim, activation_function=activation_function,
                 activation_function_out=activation_function_out, state_initializer=state_initializer)

        comb_size = hidden_dim+in_dim
        self.w_in = nn.Linear(in_features=comb_size, out_features=comb_size )
        self.w_y = nn.Linear(hidden_dim, out_dim)
        self.act_fun = activation_function
        self.act_fun_out = activation_function_out
        self.gate_update = nn.Linear(in_features=comb_size , out_features=hidden_dim)
        self.gate_relevance = nn.Linear(in_features=comb_size, out_features=comb_size)

    def forward(self, x, state=None):
        if state is None:
            satte = self.a
        ax = torch.cat((self.a, x))
        gr = torch.sigmoid(self.gate_relevance(ax))
        ar = torch.mul(gr, self.a)
        arx = torch.cat((ar, x))
        new_state = self.act_fun(self.w_in(arx))
        gu = torch.sigmoid(self.gate_update(ax))
        new_state = torch.mul(gu, new_state) + torch.mul((1-gu), self.a)
        y = self.w_y(new_state)
        y = self.act_fun_out(y)
        return y, new_state

# LSTM cell
class LSTM(Cell):
    def __init__(self, in_dim, hidden_dim, out_dim, activation_function=F.tanh,
                 activation_function_out=nn.relu, state_initializer=None):
        super(LSTM, self).__init__(in_dim, hidden_dim, out_dim, activation_function=activation_function,
                 activation_function_out=activation_function_out, state_initializer=state_initializer)
        raise NotImplementedError('LSMT not implemented')
