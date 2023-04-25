import os
import pickle
import random
import yaml
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

import numpy as np

from torchmetrics.classification import MulticlassConfusionMatrix

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def train_or_load_model(
    agent: nn.Module,
    device: torch.device,
    epochs: int,
    save_path: str,
    trainloader: torch.utils.data.DataLoader,
    logger,
    log_every: int = 2000,
    log_file_format: str = None,
    learning_rate: float = 1e-4
):
    
    if save_path and os.path.exists(save_path):
        return torch.load(save_path)
    
    agent.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(agent.parameters(), lr=learning_rate)
    global_step = 0
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            global_step += 1
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = agent(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if global_step % log_every == log_every - 1:
                if logger is None:
                        # print every `log_every` mini-batches
                        print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / log_every:.3f}')
                        running_loss = 0.0
                else:
                    logger.add_scalar("charts/loss", running_loss / log_every, global_step)
                    logger.add_scalar("charts/epochs", epoch, global_step)
                
                if save_path and log_file_format:
                    torch.save(
                        agent,
                        log_file_format.format(global_step)
                    )
    
    if save_path is not None:
        torch.save(agent, save_path)
    
    return agent


