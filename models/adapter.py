import torch
import torch.nn as nn


# Residual CLIP Adapter
class CLIPAdapter(nn.Module):
    def __init__(self, c_in, bottleneck=768):
        super(CLIPAdapter, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(c_in, bottleneck, bias=False), nn.LeakyReLU(inplace=False))
        self.fc2 = nn.Sequential(nn.Linear(bottleneck, c_in, bias=False), nn.LeakyReLU(inplace=False))

    def forward(self, x):
        x = self.fc1(x)
        y = self.fc2(x)
        return x, y
