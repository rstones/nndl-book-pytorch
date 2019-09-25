import sys
import os
import json
import datetime

import click
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from network import Network
from mnist_loaders import training_loader, test_loader

DATA_DIR = 'data'
LOG_DIR = 'logs'
SAVE_DIR = 'saved_models'

@click.command()
@click.option('--log', is_flag=True, help='Log to Tensorboard.')
@click.option('--save', is_flag=True,
            help='Save model after training. ' +
              'Note: logging must also be enabled with the --log flag to save.')
@click.argument('hparam-file') 
def run(hparam_file, log, save):
    """Trains and tests a linear neural network to classify the MNIST digits dataset. Requires a hyperparameters.json file initialise the network"""

    if save and not log:
        print('Warning: The --save flag is set without enabling logging. ' +
                'The model will not be saved!')

    # get the params from json file
    try:
        with open(hparam_file) as f:
            try:
                params = json.load(f)
            except json.JSONDecodeError as jde:
                print('Error: hyperparameters.json does not contain valid json')
                sys.exit()
    except IOError:
        print('Error: could not find hyperparameters.json')
        sys.exit()

    # extract hyperparams from config file and set defaults if missing
    bs = params.get('batchSize', 10)
    ne = params.get('numEpochs', 30)
    lr = params.get('learningRate', 0.5)
    l2r = params.get('l2Reg', 0)
    layers = params.get('layers', [784, 30, 10])

    # if log == True set up SummaryWriter and get experiment number
    # and write summary text of hyperparams to TB
    if log:
        if not os.path.isdir(LOG_DIR):
            # this is the first run and LOG_DIR hasn't been created yet
            exp_num = 1
        else:
            num_prev_runs = len(os.listdir(LOG_DIR))
            exp_num = num_prev_runs + 1

        exp_name = f'mnist_experiment_{exp_num}'
        writer = SummaryWriter(f'{LOG_DIR}/{exp_name}')
        writer.add_text(exp_name,
                        'Linear network to classify MNIST digits with ' +
                        f'layers {layers}, ' +
                        f'learning rate {lr}, ' +
                        f'L2 regularization param {l2r}, ' +
                        f'batch size {bs} ' +
                        f'and number of epochs {ne}')

    # set up model, train, test etc
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = Network(layers).to(device)
    
    print('Training model:')
    print(model)

    if log:
        writer.add_text(exp_name, f'Model architecture: \n{model}')
    
    train_load = training_loader(DATA_DIR, bs)
    test_load = test_loader(DATA_DIR, 1)
    
    #criterion = nn.MSELoss() # mean square error
    criterion = nn.BCELoss() # binary cross entropy
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=l2r)
    
    def train(epoch_idx):
    
        # set the model to training mode
        # dropout + batchnorm layers behave differently in train and test modes
        model.train()
        
        num_correct = 0
        train_loss = 0
    
        for batch_idx, (data, target) in enumerate(train_load):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            
            # flatten tensor
            data = data.view(data.shape[0], -1)
            
            #target = target.long()
            output = model(data)
    
            loss = criterion(output, target)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            
            pred = output.argmax(dim=1)
            num_correct += np.count_nonzero(torch.eq(pred,
                                                target.argmax(dim=1)))
        
        print(f'Epoch {epoch_idx} complete')

        if log:
            writer.add_scalar('train loss',
                              train_loss / len(train_load.dataset),
                              epoch)
            writer.add_scalar('train accuracy',
                              num_correct / len(train_load.dataset),
                              epoch)
    
    
    def test(epoch):
    
        # set the model to test mode
        # equivalent to model.train(mode=False)
        model.eval()
    
        test_loss = 0
        num_correct = 0
        with torch.no_grad():
            for data, target in test_load:
                data, target = data.to(device), target.to(device)
    
                data = data.view(data.shape[0], -1)
    
                output = model(data)
                test_loss += criterion(output, target).item()
    
                pred = output.argmax(dim=1)
                num_correct += 1 if pred.item() == target.argmax().item() else 0
    
        test_loss /= len(test_load.dataset)
        
        print(f'Average loss: {test_loss}, ' +
              f'Accuracy: {num_correct} / {len(test_load.dataset)}')
        
        if log:
            writer.add_scalar('test loss', test_loss, epoch)
            writer.add_scalar('test accuracy',
                              num_correct / len(test_load.dataset),
                              epoch)
    
    if log:
        writer.add_text(exp_name,
                        'Training started at ' +
                        f'{datetime.datetime.now().time()}')

    for epoch in range(ne):
        train(epoch)
        test(epoch)
    
    if log:
        writer.add_text(exp_name,
                'Training finished at ' +
                f'{datetime.datetime.now().time()}')

    # if save == True save trained model using experiment numbered file name
    if log and save:
        if not os.path.isdir(SAVE_DIR):
            os.mkdir(SAVE_DIR)
        file_path = f'{SAVE_DIR}/{exp_name}.pt'
        torch.save(model.state_dict(), file_path)
        writer.add_text(exp_name, f'Model saved at {file_path}')
    
    if log:
        writer.close()

if __name__ == '__main__':
    run()
