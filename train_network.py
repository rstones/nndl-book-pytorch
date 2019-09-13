import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

from network import Network
from mnist_loaders import training_loader, test_loader

######################
# hyperparameters
######################

lr = 3 # learning rate
bs = 10 # batch size
ne = 30 # num epochs
layers = [784, 100, 10]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Network(layers).to(device)

print(model)

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=lr)

train_loader = training_loader('data', bs)
test_loader = test_loader('data', 1)

def train(epoch_idx):

    # set the model to training mode
    # dropout + batchnorm layers behave differently in train and test modes
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        
        # flatten tensor
        data = data.view(data.shape[0], -1)

        output = model(data)

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch_idx} complete')


def test():

    # set the model to test mode
    # equivalent to model.train(mode=False)
    model.eval()

    test_loss = 0
    num_correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            data = data.view(data.shape[0], -1)

            output = model(data)
            test_loss += criterion(output, target).item()

            pred = output.argmax(dim=1)
            num_correct += 1 if pred.item() == target.argmax().item() else 0

    test_loss /= len(test_loader.dataset)
    print(f'Average loss: {test_loss}, Accuracy: {num_correct} / {len(test_loader.dataset)}')

for epoch in range(ne):
    train(epoch)
    test()


