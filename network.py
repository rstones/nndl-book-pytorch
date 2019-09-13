import numpy as np
import torch.nn as nn

class Network(nn.Module):

    def __init__(self, sizes):
        super(Network, self).__init__()
        
        layers = [self.sigmoid_layer(in_, out) for in_,out in zip(sizes, sizes[1:])]

        self.net = nn.Sequential(*layers)

    def sigmoid_layer(self, in_, out):
        return nn.Sequential(nn.Linear(in_, out), nn.Sigmoid())

    def forward(self, x):
        return self.net(x)
