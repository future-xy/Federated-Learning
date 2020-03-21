#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Name    : models.py
# Time    : 3/5/2020 3:47 PM
# Author  : Fu Yao
# Mail    : fy38607203@163.com

"""Define custom models in this file"""

import torch
from torch import nn, optim


class BostonLR(nn.Module):
    """Linear Regression model for the Boston dataset"""

    def __init__(self):
        super(BostonLR, self).__init__()
        self.fc1 = nn.Linear(13, 1)

    def forward(self, x):
        return self.fc1(x).flatten()


class MnistDNN(nn.Module):
    """Deep neural network for the Mnist dataset"""

    def __init__(self):
        super(MnistDNN, self).__init__()
        self.fc1 = nn.Linear(784, 200)
        self.fc2 = nn.Linear(200, 200)
        self.out = nn.Linear(200, 10)
        self.re = nn.ReLU()

    def forward(self, x):
        y = self.re(self.fc1(x))
        y = self.re(self.fc2(y))
        return self.out(y)
