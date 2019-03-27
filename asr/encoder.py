import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import RNN, GRU, LSTM

class GRUEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, out_dim, num_layers=1, act_out=nn.LeakyReLU):
        super(GRUEncoder, self).__init__()
        self.gru = GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.fc1 = nn.Linear(in_features=hidden_size, out_features=256)
        self.fc_out = nn.Linear(in_features=256, out_features=out_dim)
        self.act_fun = nn.functional.relu
        self.latent_dim = out_dim
        self.act_out = act_out()



    def forward(self, x):
        out, h_n = self.gru(x)
        f1 = nn.functional.leaky_relu(self.fc1(out[:, -1, :]))
        embedding = self.act_out(self.fc_out(f1))  # do fully connected layers make sense here? mu and log_var should do this?
        # embedding = out[-1, :, :]  # output of recurrent unit for last element of sequence
        return embedding, h_n


class LSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, out_dim, num_layers=1, act_out=nn.LeakyReLU):
        super(LSTMEncoder, self).__init__()
        self.cell = LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.fc1 = nn.Linear(in_features=hidden_size, out_features=256)
        self.fc_out = nn.Linear(in_features=256, out_features=out_dim)
        self.act_fun = nn.functional.relu
        self.latent_dim = out_dim
        self.act_out = act_out()


    def forward(self, x):
        out, (h_n, c_n) = self.cell(x)
        f1 = nn.functional.leaky_relu(self.fc1(out[:, -1, :]))
        embedding = self.act_out(
            self.fc_out(f1))  # do fully connected layers make sense here? mu and log_var should do this?
        # embedding = out[-1, :, :]  # output of recurrent unit for last element of sequence
        return embedding, (h_n, c_n)
