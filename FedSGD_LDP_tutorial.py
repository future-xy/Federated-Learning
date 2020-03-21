#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Name    : FedSGD_LDP_tutorial.py
# Time    : 3/16/2020 9:37 PM
# Author  : Fu Yao
# Mail    : fy38607203@163.com

"""
This tutorial is about how to construct and test FedAvg + Local DP model
"""

import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import numpy as np

from tests.models import MnistDNN
from tests.settings import args, device
from FLsim.federated_learning import SerialFedAvg

torch.manual_seed(args.seed)
np.random.seed(args.seed)

args.test_batch_size = 2000
args.client_count = 3000


def central():
    """The first part is central training model"""
    args.batch_size = 20
    args.epochs = 5
    train_set = datasets.MNIST(root="./data", train=True, transform=transforms.ToTensor(), download=True)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)

    print("Central training")
    model = MnistDNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        losses = 0
        for i, (X, y) in enumerate(train_loader):
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_ = model(X.flatten(start_dim=1))
            loss = criterion(y_, y)
            loss.backward()
            optimizer.step()
            losses += loss.item()
        print(losses / len(train_loader))
    print("END")


def FedAvg():
    """This part is FedAvg model"""
    args.epochs = 5
    trans = [transforms.ToTensor(),
             transforms.Lambda(lambda x: x.reshape(-1))]
    train_set = datasets.MNIST(root="./data", train=True, transform=transforms.Compose(trans), download=True)
    FL = SerialFedAvg(MnistDNN, device, args.client_count)
    # IID dataset
    clients = [i % args.client_count for i in range(len(train_set))]
    FL.federated_data(train_set, clients, args.batch_size)

    print("FedAvg")
    model = MnistDNN().to(device)
    state = model.state_dict().copy()
    for epoch in range(args.epochs):
        state, loss = FL.global_update(state, lr=args.lr, E=args.E)
        print("Epoch {}\tLoss: {}".format(epoch, loss))
    model.load_state_dict(state.copy())


if __name__ == '__main__':
    # central()
    FedAvg()
