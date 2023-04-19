import os
import random
import yaml
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchmetrics.classification import MulticlassConfusionMatrix
from ccgm.common.envs.sl.cifar10.config import ROOT_DIR

from ccgm.common.envs.sl.cifar10.net import Net


seed = 1234
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)


def train_model(save_path: str = 'cifar10_net.pth'):
    save_path = os.path.join(ROOT_DIR, save_path)
    
    if os.path.exists(save_path):
        return torch.load(save_path)
    
    net = Net()
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=1e-4)

    for epoch in range(200):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    torch.save(net, save_path)
    return net


def compute_confusion_matrix(net: nn.Module, save_path: str = 'cifar10_cfm.npy'):
    save_path = os.path.join(ROOT_DIR, save_path)

    if os.path.exists(save_path):
        return np.loadtxt(save_path)
    
    confusion_matrix = MulticlassConfusionMatrix(num_classes=10)
    confusion_matrix.to(device)
    with torch.no_grad():
        for data in trainloader:
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
    confusion_matrix: np.array,
    save_path: str = 'players.json',
    num_players: int = 6,
): 
    save_path = os.path.join(ROOT_DIR, save_path)

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
    while len(players_set) < num_players:
        players_set.update(most_confused_pairs[:, idx])
        idx += 1
    
    players = {int(idx): classes[idx] for idx in players_set}
    with open(save_path, mode='w+') as f:
        yaml.dump(players, f)

    return players_set


def main():
    model = train_model()
    confusion_matrix = compute_confusion_matrix(model)
    compute_treachorus_pairs(confusion_matrix)
    

if __name__ == '__main__':
    main()
