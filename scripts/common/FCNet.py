import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import flatten

class FCNet(nn.Module):
    def __init__(self, layers, in_size=13, dropout=None):
        super(FCNet, self).__init__()

        self.fc0 = nn.Linear(in_size, layers[0])
        self.fc1 = nn.Linear(layers[0], layers[1])
        self.fc2 = nn.Linear(layers[1], layers[2])
        self.fc_final = nn.Linear(layers[-1], 1)

        self.dropout = dropout
        self.use_dropout = True if dropout else False
        if self.use_dropout:
            print(f'Using dropout: {dropout}')
            self.dropout0 = nn.Dropout(p=dropout)
            self.dropout1 = nn.Dropout(p=dropout)
            self.dropout2 = nn.Dropout(p=dropout)
        else:
            self.dropout0 = nn.Identity()
            self.dropout1 = nn.Identity()
            self.dropout2 = nn.Identity()

    def forward(self, x, targets=None, epoch=None):
        x = self.dropout0(F.relu(self.fc0(x)))
        x = self.dropout1(F.relu(self.fc1(x)))
        x = self.dropout2(F.relu(self.fc2(x)))
        x = self.fc_final(x)

        return x
