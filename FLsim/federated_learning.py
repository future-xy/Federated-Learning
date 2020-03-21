#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Name    : federated_learning.py
# Time    : 3/5/2020 8:11 PM
# Author  : Fu Yao
# Mail    : fy38607203@163.com

import torch
import numpy as np
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

from abc import ABCMeta, abstractmethod
from FLsim.federated_data import FederatedDataset


class FederatedLearning(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, Model, device, client_count, optimizer, criterion):
        """init"""
        pass

    def _fed_avg(self):
        """Execute FedAvg algorithm"""
        pass

    def _client_update(self, client_id, lr, E):
        """Update the model on client"""
        pass

    def _send(self, state):
        """Duplicate the central model to the client"""
        pass

    def global_update(self, state, lr, E):
        """Execute one round of serial global update"""
        pass

    def federated_data(self, dataset, clients, batch_size):
        """Construct the federated dataset"""
        pass


class FLBase(FederatedLearning):
    """This class is the base of FL and should not be used"""

    def __init__(self, Model, device, client_count, optimizer, criterion):
        super(FLBase, self).__init__(Model, device, client_count, optimizer, criterion)
        self.Model = Model
        self.device = device
        self.client_count = client_count
        self.models = [Model().to(self.device) for _ in range(self.client_count)]
        self.optimizer = optimizer
        self.criterion = criterion

    def _fed_avg(self):
        """Execute FedAvg algorithm"""
        self.weights = np.array(self.weights) / sum(self.weights)
        with torch.no_grad():
            clients_updates = [model.state_dict() for model in self.models]
            new_parameters = clients_updates[0].copy()
            for name in new_parameters:
                new_parameters[name] = torch.zeros(new_parameters[name].shape).to(self.device)
            for idx, parameter in enumerate(clients_updates):
                for name in new_parameters:
                    new_parameters[name] += parameter[name] * self.weights[idx]
        return new_parameters.copy()

    def federated_data(self, dataset, clients, batch_size):
        """Construct the federated dataset"""
        if self.client_count > len(dataset):
            return "ERROR"
        self.client_dataloader = [
            DataLoader(FederatedDataset(dataset, clients, client_id), batch_size=batch_size, shuffle=True)
            for client_id in range(self.client_count)]


class SerialFedAvg(FLBase):
    def __init__(self, Model, device, client_count, optimizer, criterion):
        super(SerialFedAvg, self).__init__(Model, device, client_count, optimizer, criterion)
        self.Model = Model
        self.device = device
        self.client_count = client_count
        self.models = [Model().to(self.device) for _ in range(self.client_count)]
        self.optimizer = optimizer
        self.criterion = criterion

    def _send(self, state):
        """Duplicate the central model to the client"""
        for model in self.models:
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    parameter.data = state[name].clone()
        self.weights = [0] * self.client_count
        self.losses = [0] * self.client_count

    def _client_update(self, client_id, lr, E):
        """Update the model on client"""
        model = self.models[client_id]
        optimizer = self.optimizer(model.parameters(), lr=lr)
        criterion = self.criterion(reduction="sum")
        dataloader = self.client_dataloader[client_id]
        weight = 0
        losses = 0
        for e in range(E):
            for data, target in dataloader:
                data,target=data.to(self.device),target.to(self.device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                # Record loss
                losses += loss.item()
                weight += len(data)
                optimizer.step()
        with torch.no_grad():
            self.weights[client_id] = weight / E
            self.losses[client_id] = losses / E / weight

    def global_update(self, state, lr, E=1):
        """Execute one round of serial global update"""
        self._send(state)
        for i in range(self.client_count):
            self._client_update(i, lr, E)
        return self._fed_avg(), sum(self.losses) / self.client_count


class ParallelFedAvg(FLBase):
    def __init__(self, Model, device, client_count, optimizer, criterion, manager):
        super(ParallelFedAvg, self).__init__(Model, device, client_count, optimizer, criterion)
        self.Model = Model
        self.device = device
        self.client_count = client_count
        self.models = [Model().to(self.device) for _ in range(self.client_count)]
        self.queue = manager.Queue()
        self.criterion = criterion
        self.optimizer = optimizer

    def _send(self, state):
        """Duplicate the central model to the client"""
        for model in self.models:
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    parameter.data = state[name].clone()
        self.weights = [0] * self.client_count
        self.losses = [0] * self.client_count

    def _recv(self):
        for _ in range(self.client_count):
            (i, weight, loss) = self.queue.get()
            self.weights[i] = weight
            self.losses[i] = loss

    def _client_update(self, client_id, lr, E):
        """Update the model on client"""
        model = self.models[client_id]
        optimizer = self.optimizer(model.parameters(), lr=lr)
        criterion = self.criterion(reduction="sum")
        dataloader = self.client_dataloader[client_id]
        weight = 0
        losses = 0
        for e in range(E):
            for data, target in dataloader:
                data,target=data.to(self.device),target.to(self.device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                # Record loss
                losses += loss.item()
                weight += len(data)
                optimizer.step()
        with torch.no_grad():
            self.queue.put((client_id, weight / E, losses / E / weight))

    def global_update(self, state, lr, E=1):
        """Execute one round of serial global update"""
        self._send(state)
        mp.spawn(self._client_update, (lr, E), nprocs=self.client_count)
        self._recv()
        return self._fed_avg(), sum(self.losses) / self.client_count


class FedSGD_LocalDP(FederatedLearning):
    def __init__(self):
        super(FedSGD_LocalDP, self).__init__()
        pass
