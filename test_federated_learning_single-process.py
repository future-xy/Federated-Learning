#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Name    : test_federated_learning_single-process.py
# Time    : 2020/3/3 13:55
# Author  : Fu Yao
# Mail    : fy38607203@163.com

from torch import nn
import numpy as np

from model import Model
from training_settings import device, args
from federated_data import construct_federated_dataset
from federated_learning import FederatedLearning
from utils import *

torch.manual_seed(args.seed)
np.random.seed(args.seed)

if __name__ == '__main__':
    # Load Boston dataset
    train_loader, test_loader = get_boston_dataset(device, 0.9, args.batch_size, args.test_batch_size)
    # Construct the federated dataset
    clients_data = construct_federated_dataset(train_loader, args.client_count)

    record_loss = []
    FL = FederatedLearning(Model, device)
    model = Model().to(device)
    state = model.state_dict().copy()
    for epoch in range(args.epochs):
        state, loss = FL.global_update_single_process(clients_data, state, lr=args.lr, E=args.E)
        print("Epoch {}\tLoss: {}".format(epoch, loss))
        record_loss.append(loss)
    # Evaluate the final model
    model.load_state_dict(state.copy())
    criterion = nn.MSELoss(reduction="sum")
    loss = eval(model, test_loader, criterion)
    print("Loss on the test dataset: {}".format(loss))
    np.save("Federated learning loss(E={})".format(args.E), record_loss)
