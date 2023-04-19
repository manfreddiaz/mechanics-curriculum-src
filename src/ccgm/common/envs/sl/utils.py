import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

from torchmetrics.classification import MulticlassConfusionMatrix

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def train_or_load_model(
    model: nn.Module,
    device: torch.device,
    epochs: int,
    save_path: str,
    trainloader: torch.utils.data.DataLoader
):
    
    if os.path.exists(save_path):
        return torch.load(save_path)
    
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    torch.save(model, save_path)
    return model

def compute_confusion_matrix(
    net: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    save_path: str
):

    if os.path.exists(save_path):
        return np.loadtxt(save_path)
    
    confusion_matrix = MulticlassConfusionMatrix(num_classes=10)
    confusion_matrix.to(device)
    with torch.no_grad():
        for data in dataloader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = net(inputs)
            confusion_matrix(outputs, labels)

    conf_mat = confusion_matrix.compute()
    cfm = conf_mat.cpu().numpy()
    np.savetxt(save_path, cfm)
    
    return cfm


def compute_treachorus_pairs(
    player_ids: list[str],
    max_players: int,
    confusion_matrix: np.array,
    save_path: str,

): 

    confusion_matrix[
        np.diag_indices_from(confusion_matrix)] = 0.0

    most_confused_pairs = (
        np.arange(confusion_matrix.shape[0]), 
        np.argmax(confusion_matrix, axis=0)
    )
    most_confused_values = confusion_matrix[most_confused_pairs]
    most_confused_pairs = np.array(most_confused_pairs)
    confusion_order = np.argsort(most_confused_values)[::-1]
    most_confused_pairs = most_confused_pairs[:, confusion_order]

    players_set = set()
    idx = 0
    while len(players_set) < max_players:
        players_set.update(most_confused_pairs[:, idx])
        idx += 1
    
    players = {int(idx): player_ids[idx] for idx in players_set}
    with open(save_path, mode='w+') as f:
        yaml.dump(players, f)
    
    pairings_file = os.path.join(
        os.path.dirname(save_path),
        "pairings.npy"
    )
    np.savetxt(pairings_file, most_confused_pairs)

    return players_set
