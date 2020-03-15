#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Name    : settings.py
# Time    : 3/5/2020 3:58 PM
# Author  : Fu Yao
# Mail    : fy38607203@163.com

"""Training settings and utils"""

import torch


class Arguments():
    def __init__(self):
        self.batch_size = 10
        self.test_batch_size = 16
        self.epochs = 10
        self.lr = 0.01
        self.seed = 0
        self.client_count = 10
        self.E = 1


args = Arguments()
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
