#!/usr/bin/env python

import torch
import torch.nn as nn

class RnnLstm(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_int_dim, output_dim, device):
        super(RnnLstm, self).__init__()

        self.layer_dim = layer_dim                              # N° of LSTM layers
        self.hidden_dim = hidden_dim                            # N° of neurons of LSTM hidden layer
        self.device = device
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.silu = nn.SiLU()                                   # activation function
        self.int_fc = nn.Linear(hidden_dim, output_int_dim)     # 1° lin layer
        self.fc = nn.Linear(output_int_dim, output_dim)         # 2° lin layer
        # fully connected layer to convert the RNN output into desired output shape

    def forward(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(self.device)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(self.device)
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))  # h_0,c_0 = zeros by default
        out = self.silu(out)
        out = self.int_fc(out[:, -1, :])         # the last set of feature for each elem of the batch
        out = self.silu(out)
        out = self.fc(out)
        return out, (hn, cn)
