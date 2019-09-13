import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image

class Flatten(object):
    """A dataset transform to flatten the MNIST images"""

    def __call__(self, sample):
        return sample.view(1, -1)
    
class TargetToOutputArray(object):
    """Converts integer target into array to compare with network output"""

    def __call__(self, target):
        new_target = torch.zeros(10)
        new_target[target] = 1.
        return new_target
 

def training_loader(data_dir, batch_size):
    train_data = datasets.MNIST(data_dir, train=True, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                           # Flatten()
                        ]),
                        target_transform=TargetToOutputArray())
    return DataLoader(train_data, batch_size=batch_size, shuffle=True)

def test_loader(data_dir, batch_size):
    test_data = datasets.MNIST(data_dir, train=False, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            #Flatten()
                        ]),
                        target_transform=TargetToOutputArray())
    return DataLoader(test_data, batch_size=batch_size, shuffle=True)



