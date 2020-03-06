#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Name    : utils.py
# Time    : 3/5/2020 7:03 PM
# Author  : Fu Yao
# Mail    : fy38607203@163.com

import torch
from torch.utils.data import TensorDataset, DataLoader

from sklearn.datasets import load_boston


def get_boston_dataset(training_data=0.9, batch_size=10, test_batch_size=20):
    """Load Boston dataset and convert to Torch tensor"""
    X, y = load_boston(return_X_y=True)
    # Normalize
    X = (X - X.mean(axis=0, keepdims=True)) / X.std(axis=0, keepdims=True)
    # Convert Numpy array to Torch tensor
    X = torch.from_numpy(X).type(torch.float)
    y = torch.from_numpy(y).type(torch.float)

    # Divide the dataset into train and dev
    train_count = int(training_data * len(X))

    # Data loader
    train_loader = DataLoader(TensorDataset(X[:train_count], y[:train_count]), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X[train_count:], y[train_count:]), batch_size=test_batch_size, shuffle=True)
    return train_loader, test_loader


def eval(device, model, datas, criterion):
    """Eval the model"""
    losses = 0
    model.eval()
    with torch.no_grad():
        for data, target in datas:
            output = model(data.to(device)).flatten()
            losses += criterion(output.flatten(), target.to(device)).item()
    return losses / len(datas.dataset)
