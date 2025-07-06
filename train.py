import torch
from torch.utils.data import DataLoader
from typing import Optional
from models.ctm import ContinousThoughtMachine
from utils import calculate_accuracy
from losses import loss_mnist_

# Hyperparameters
LR = 0.0001








def train(model: ContinousThoughtMachine, train_loader: DataLoader,  num_iteration: int, test_loader: Optional[DataLoader], device, track: bool = False):

    optim = torch.optim.AdamW(model.parameters(), lr = LR)

    model.train()
    for step in range(num_iteration):
        inputs, targets = next(iter(train_loader))
        inputs, targets = inputs.to(device), targets.to(device) # Offloading the batch to GPU
        predictions, certainities, _ = model(inputs, track=track)
        train_loss, where_most_certain = loss_mnist_(predictions, certainities, targets)
        train_accuracy = calculate_accuracy(predictions, targets, where_most_certain)

        optim.zero_grad()
        train_loss.backward()
        optim.step()
