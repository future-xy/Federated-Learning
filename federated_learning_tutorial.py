#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Name    : federated_learning_tutorial.py
# Time    : 2020/3/3 13:55
# Author  : Fu Yao
# Mail    : fy38607203@163.com

"""
This file is to show how to use this framework in general ML scenarios.

Because this model (Linear Regression) is super simple, multi-process FL
is slower then single process.

Multi-process FL, however, is useful in complex model,
when client's updates cost more time than creating process.
"""

from torch import nn
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import time
from sklearn.datasets import load_boston

from tests.models import Model
from tests.settings import device, args
from FLsim.federated_learning import SerialFL, ParallelFL
from tests.utils import *


def single_process(train_data, test_loader):
    # Fix random seed then the result should be same
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # Construct the SERIAL federated model
    FL = SerialFL(Model, device, args.client_count)
    """
    A list of client id, whose length equals to training data count, 
    should be given, which means the owner of each data.

    If you want to test your algorithm on Non-IID dataset, 
    you need to change the clients below into any distribution.
    """
    # IID dataset
    clients = [i % args.client_count for i in range(len(train_data))]
    # Construct the federated dataset
    FL.federated_data(train_data, clients, args.batch_size)

    # Single process FL
    start = time.time()
    record_loss = []
    model = Model().to(device)
    state = model.state_dict().copy()
    for epoch in range(args.epochs):
        state, loss = FL.global_update(state, lr=args.lr, E=args.E)
        print("Epoch {}\tLoss: {}".format(epoch, loss))
        record_loss.append(loss)

    # Evaluate the trained model
    model.load_state_dict(state.copy())
    print("Traing time: {}".format(time.time() - start))
    criterion = nn.MSELoss(reduction="sum")
    loss = eval(device, model, test_loader, criterion)
    print("Loss on the test dataset: {}".format(loss))


def multi_process(train_data, test_loader):
    # Test multi-process FL
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    """manager is required"""
    manager = mp.Manager()
    FL = ParallelFL(Model, device, args.client_count, manager)
    """
    A list of client id, whose length equals to training data count, 
    should be given, which means the owner of each data.

    If you want to test your algorithm on Non-IID dataset, 
    you need to change the clients below into any distribution.
    """
    # IID dataset
    clients = [i % args.client_count for i in range(len(train_data))]
    FL.federated_data(train_data, clients, args.batch_size)
    # Start
    start = time.time()
    model = Model().to(device)
    state = model.state_dict().copy()
    for epoch in range(args.epochs):
        state, loss = FL.global_update(state, args.lr, args.E)
        print("Epoch {}\tLoss: {}".format(epoch, loss))

    # Evaluate the trained model
    model.load_state_dict(state.copy())
    print("Traing time: {}".format(time.time() - start))
    criterion = nn.MSELoss(reduction="sum")
    loss = eval(device, model, test_loader, criterion)
    print("Loss on the test dataset: {}".format(loss))


if __name__ == '__main__':
    """Load any dataset"""
    # Load Boston dataset
    X, y = load_boston(return_X_y=True)
    # Normalize
    X = (X - X.mean(axis=0, keepdims=True)) / X.std(axis=0, keepdims=True)
    # Convert Numpy array to Torch tensor
    X = torch.from_numpy(X).type(torch.float)
    y = torch.from_numpy(y).type(torch.float)
    training_data_count = 500
    """You need to construct a dataset whose type is torch.utils.data.Dataset"""
    train_data = TensorDataset(X[:training_data_count], y[:training_data_count])
    test_loader = DataLoader(TensorDataset(X[training_data_count:], y[training_data_count:]),
                             batch_size=args.batch_size, shuffle=True)

    # Compare single and multi process FL
    single_process(train_data, test_loader)
