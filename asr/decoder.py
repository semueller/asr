import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import RNN, GRU, LSTM

class GRUDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, out_dim, num_layers=1):
        super(GRUDecoder, self).__init__()
        self.fc_h0 = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.gru = GRU(input_size=out_dim, hidden_size=hidden_size, batch_first=True)
        self.fc_out = nn.Linear(in_features=hidden_size, out_features=out_dim)

    def forward(self, x, seq_len):
        # t = 0
        h_t = nn.functional.tanh(self.fc_h0(x))  # compute initial state from embedding
        x_t = nn.functional.leaky_relu(self.fc_out(h_t), negative_slope=0.04)
        out = [x_t]
        for i in range(seq_len-1):
            _, h_t = self.gru(torch.unsqueeze(x_t, 1), hx=torch.unsqueeze(h_t, 0))
            h_t = torch.squeeze(h_t, 0)
            x_t = nn.functional.leaky_relu(self.fc_out(h_t), negative_slope=0.04)
            out.append(x_t)
        out = torch.stack(out, dim=1)
        return out

