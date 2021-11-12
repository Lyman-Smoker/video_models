from typing import Tuple
import torch
import torch.nn as nn


class C3D(nn.Module):
    def __init__(self):
        super(C3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=3, out_channels=64 ,kernel_size=(3,3,3), padding=(1,1,1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(in_channels=64, out_channels=128 ,kernel_size=(3,3,3), padding=(1,1,1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2,2,2), stride=(2, 2, 2))

        self.conv3 = nn.Sequential(
            nn.Conv3d(in_channels=128, out_channels=256 ,kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1)),
            nn.Conv3d(in_channels=256, out_channels=256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        )
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4 = nn.Sequential(
            nn.Conv3d(in_channels=256, out_channels=512, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.Conv3d(in_channels=512, out_channels=512, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        )
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5 = nn.Sequential(
            nn.Conv3d(in_channels=512, out_channels=512, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.Conv3d(in_channels=512, out_channels=512, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        )
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))

    def forward(self, x):
        out1 = self.pool1(self.conv1(x))
        out2 = self.pool2(self.conv2(out1))
        out3 = self.pool3(self.conv3(out2))
        out4 = self.pool4(self.conv4(out3))
        out5 = self.pool5(self.conv5(out4))

        out = out5.view(-1, 8192)
        return out

if __name__ == '__main__':
    input = torch.randn(1, 3, 16, 112, 112)
    c3d_net = C3D()
    output = c3d_net(input)
    print(output.shape)