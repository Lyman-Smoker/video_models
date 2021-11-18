import torch.nn as nn
import torch
import numpy as np

class Regressor(nn.Module):
    def __init__(self):
        super(Regressor, self).__init__()
        self.fc1 = nn.Linear(8192, 4096)
        self.relu = nn.LeakyReLU()
        # self.fc2 = nn.Linear(4096, 2048)
        self.fc_final_score = nn.Linear(4096, 1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        # x = self.relu(self.fc2(x))
        final_score = self.fc_final_score(x)
        return final_score

if __name__ == '__main__':
    rgs = Regressor()
    x = torch.randn(3, 8192)
    print(rgs(x).shape)