#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Name    : FedAvg_LDP_tutorial.py
# Time    : 3/16/2020 9:37 PM
# Author  : Fu Yao
# Mail    : fy38607203@163.com

"""
This tutorial is about how to construct and test FedAvg + Local DP model
"""

import numpy as np
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from tests.models import MnistDNN
from tests.settings import args, device
from tests.utils import *

torch.manual_seed(args.seed)
np.random.seed(args.seed)

args.test_batch_size = 2000
args.client_count = 3000
args.lr = 0.1

if __name__ == '__main__':
    """The first part is central training model"""
    args.batch_size = 20
    args.epochs = 5
    train_set = datasets.MNIST(root="./data", train=True, transform=transforms.ToTensor(), download=True)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_set = datasets.MNIST(root="./data", train=False, transform=transforms.ToTensor(), download=True)
    test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=True)

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
