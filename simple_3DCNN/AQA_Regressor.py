import torch.nn as nn
import torch
import numpy as np

class Regressor(nn.Module):
    def __init__(self):
        super(Regressor, self).__init__()
        self.fc = nn.Linear(8192, 4096)
        self.relu = nn.ReLU()
        self.fc_final_score = nn.Linear(4096, 1)

    def forward(self, x):
        x = self.relu(self.fc(x))
        final_score = self.fc_final_score(x)
        return x