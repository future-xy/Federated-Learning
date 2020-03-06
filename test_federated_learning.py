#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Name    : test_federated_learning.py
# Time    : 2020/3/3 13:55
# Author  : Fu Yao
# Mail    : fy38607203@163.com

"""This code is to verify the multi-process FL and
compare the multi-process training time with single process.

Because this model is super simple, multi-process FL is slower
then single process.

Multi-process FL, however, is useful in complex model,
when client's updates cost more time than creating process.
"""

from torch import nn
import numpy as np
import time

from model import Model
from training_settings import device, args
from federated_data import construct_federated_dataset
from federated_learning import FederatedLearning
from utils import *

if __name__ == '__main__':
    # Load Boston dataset
    train_loader, test_loader = get_boston_dataset(0.9, args.batch_size, args.test_batch_size)
    # Construct the federated dataset
    clients_data = construct_federated_dataset(train_loader, args.client_count)

    # # Compare single and multi process FL
    # # Fix random seed then the result should be same
    # torch.manual_seed(args.seed)
    # np.random.seed(args.seed)
    # # Single process FL
    # start = time.time()
    # record_loss = []
    # FL = FederatedLearning(Model, device)
    # model = Model().to(device)
    # state = model.state_dict().copy()
    # for epoch in range(args.epochs):
    #     state, loss = FL.serial_global_update(clients_data, state, lr=args.lr, E=args.E)
    #     print("Epoch {}\tLoss: {}".format(epoch, loss))
    #     record_loss.append(loss)
    # # Evaluate the final model
    # model.load_state_dict(state.copy())
    # print("Traing time: {}".format(time.time() - start))
    # criterion = nn.MSELoss(reduction="sum")
    # loss = eval(device, model, test_loader, criterion)
    # print("Loss on the test dataset: {}".format(loss))
    # # np.save("Federated learning loss(E={})".format(args.E), record_loss)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # Test multi-process FL
    start = time.time()
    FL = FederatedLearning(Model, device)
    model = Model()
    state = model.state_dict().copy()
    for epoch in range(args.epochs):
        state, loss = FL.parallel_global_update(clients_data, state, lr=args.lr, E=args.E)
        print("Epoch {}\tLoss: {}".format(epoch, loss))
    # Evaluate the final model
    model.load_state_dict(state.copy())
    print("Traing time: {}".format(time.time() - start))
    criterion = nn.MSELoss(reduction="sum")
    loss = eval(device, model, test_loader, criterion)
    print("Loss on the test dataset: {}".format(loss))
