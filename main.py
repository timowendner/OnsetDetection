import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import pickle as pkl
import json

import argparse
import datetime
import time
import os
import matplotlib.pyplot as plt

from utils import create_model, save_model
from dataloader import OnsetDataset
from model import UNet


@torch.no_grad()
def test_network(model, dataset):
    model.eval()
    for idx in range(len(dataset)):
        full = dataset.dataset.getFull(idx)
        prediction_full = np.array([])
        targets_full = np.array([])
        input_full = np.array([])
        for model_input, targets in zip(*full):
            model_input = model_input.unsqueeze(0)
            targets = targets.unsqueeze(0)
            prediction = model(model_input)
            targets = targets.view(targets.size(0), -1).cpu().numpy()
            prediction = prediction.view(prediction.size(0), -1).cpu().numpy()
            model_input = model_input.cpu().numpy()

            # calculate the MSE
            n = model_input.shape[2] // 4
            prediction_full = np.append(prediction_full, prediction[0, n:3*n])
            targets_full = np.append(targets_full, targets[0, n:3*n])
            input_full = np.append(input_full, model_input[0, 0, n:3*n])

    prediction_full = prediction_full[model_input.shape[2] // 2:]
    targets_full = targets_full[model_input.shape[2] // 2:]
    input_full = input_full[model_input.shape[2] // 2:]

    # with open('datatomesswith.p', 'wb') as f:
    #     pkl.dump((prediction_full, targets_full, input_full), f)

    plt.figure(figsize=(10, 3))
    plt.plot(input_full, label='model input')
    plt.plot(targets_full, label='target')
    plt.plot(prediction_full, label='prediction')
    plt.legend()
    plt.show()

    # Calculate overall MSE
    return 1


def train_network(model, config, optimizer):
    # create the dataset
    dataset = OnsetDataset(config, config.device)

    # split the dataset into train and test
    train_size = int(len(dataset) * config.train_split)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(
        dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Train the model
    model.train()
    total_step = len(train_loader)
    mse = torch.nn.MSELoss()

    start_time = time.time()
    for epoch in range(config.num_epochs):
        # print the epoch and current time
        time_now = datetime.datetime.now()
        time_now = time_now.strftime("%H:%M")
        lr = optimizer.param_groups[0]['lr']
        print(
            f"Start Epoch: {epoch + 1}/{config.num_epochs}   {time_now}   (lr: {lr})")

        # loop through the training loader
        for i, (model_input, targets) in enumerate(train_loader):
            # Forward pass
            outputs = model(model_input)
            loss = mse(outputs, targets)

            # calculate gradients
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 50 == 0:
                print(f'Epoch [{epoch + 1}/{config.num_epochs}]',
                      f'Step [{i + 1}/{total_step}]',
                      f'Loss: {loss.item():.4f}')

        # add the number of epochs
        config.current_epoch += 1

        # save the model if enough time has passed
        if abs(time.time() - start_time) >= config.save_time or epoch == config.num_epochs - 1:
            test_error = test_network(model, test_dataset)
            print(f'The test mse is: {test_error}')
            save_model(model, optimizer, config)
            start_time = time.time()

    return model, config, optimizer


def main():
    # Load JSON config file
    with open(args.config_path) as f:
        config_json = json.load(f)

    # create the config file
    class Config:
        pass
    config = Config()
    for key, data in config_json.items():
        setattr(config, key, data)
    config.current_epoch = 0

    # set the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config.device = device

    # create the model
    model, config, optimizer = create_model(config, load=args.load, lr=args.lr)
    model = model.to(device)

    # print the number of trainable parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(', '.join([
        f'Number of trainable parameters: {num_params:,}',
        f'with epoch {config.current_epoch}',
    ]))

    # train the network
    if args.train:
        train_network(model, config, optimizer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Diffusion Model')
    parser.add_argument('--train', action='store_true',
                        help='Train the model')
    parser.add_argument('--config_path', type=str,
                        help='Path to the configuration file')
    parser.add_argument('--load', action='store_true',
                        help='load a model')
    parser.add_argument('--lr', type=float, default=False,
                        help='change the learning rate')
    args = parser.parse_args()

    main()
