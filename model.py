#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Name    : model.py
# Time    : 3/5/2020 3:47 PM
# Author  : Fu Yao
# Mail    : fy38607203@163.com

"""Define custom model in this file"""

import torch
from torch import nn, optim


class Model(nn.Module):
    """Linear Regression model for the Boston dataset"""

    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(13, 1)

    def forward(self, x):
        return self.fc1(x)
