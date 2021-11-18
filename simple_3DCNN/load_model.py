import torch
from C3D_model import C3D

inputs = torch.rand(1, 3, 16, 112, 112)     # [batch x channels x num_clips x W x H]
net = C3D(num_classes=101, pretrained=True, pretrained_model_path='./c3d-pretrained.pth')
outputs = net.forward(inputs)
print(outputs.size())