#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Name    : test_central_training.py
# Time    : 3/5/2020 4:10 PM
# Author  : Fu Yao
# Mail    : fy38607203@163.com

"""Training on the Boston dataset"""

import numpy as np
from torch import nn, optim

from tests.models import Model
from tests.settings import args, device
from tests.utils import *

torch.manual_seed(args.seed)
np.random.seed(args.seed)

if __name__ == '__main__':
    # Load Boston dataset
    train_loader, test_loader = get_boston_dataset(0.9, args.batch_size, args.test_batch_size)

    model = Model().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    record_train_loss = []
    for epoch in range(args.epochs):
        losses = 0
        for idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data.to(device)).flatten()
            loss = criterion(output, target.to(device))
            loss.backward()
            losses += loss.item()
            optimizer.step()
            print("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                epoch, idx * args.batch_size, len(train_loader) * args.batch_size,
                       100. * idx / len(train_loader), loss
            ))
        print("Train Epoch: {}\t Mean loss: {:.6f}".format(epoch, losses / len(train_loader)))
        record_train_loss.append(losses / len(train_loader))
    print("Evaluating...")
    criterion = nn.MSELoss(reduction="sum")
    losses = eval(device, model, test_loader, criterion)
    print("Test loss: {}".format(losses / len(test_loader)))
    # np.save("Central training loss", record_train_loss)
