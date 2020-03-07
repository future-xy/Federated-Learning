#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Name    : federated_learning.py
# Time    : 3/5/2020 8:11 PM
# Author  : Fu Yao
# Mail    : fy38607203@163.com

import torch
import numpy as np
from torch import optim, nn
import torch.multiprocessing as mp

from abc import ABCMeta, abstractmethod


class FederatedLearning(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, Model, device, client_count):
        """init"""
        pass

    def _fed_avg(self, clients_updates, weights):
        """Execute FedAvg algorithm"""
        pass

    def _client_update(self, client_id, model, lr, E):
        """Update the model on client"""
        pass

    def _send(self, state):
        """Duplicate the central model to the client"""
        pass

    def global_update(self, state, lr, E):
        """Execute one round of serial global update"""
        pass

    def federated_data(self, dataset):
        """Construct the federated dataset"""
        pass


class SerialFL(FederatedLearning):
    def __init__(self, Model, device, client_count):
        super(SerialFL, self).__init__(Model, device, client_count)
        self.Model = Model
        self.device = device
        self.client_count = client_count

    def _fed_avg(self, clients_updates, weights):
        """Execute FedAvg algorithm"""
        with torch.no_grad():
            new_parameters = clients_updates[0].copy()
            weights = np.array(weights) / sum(weights)
            for name in new_parameters:
                new_parameters[name] = torch.zeros(new_parameters[name].shape).to(self.device)
            for idx, parameter in enumerate(clients_updates):
                for name in new_parameters:
                    new_parameters[name] += parameter[name] * weights[idx]
        return new_parameters.copy()

    def _client_update(self, client_id, model, lr, E):
        """Update the model on client"""
        optimizer = optim.SGD(model.parameters(), lr=lr)
        criterion = nn.MSELoss(reduction="sum")
        weight = 0
        losses = 0
        for e in range(E):
            for data, target in self.clients_data[client_id]:
                optimizer.zero_grad()
                output = model(data).flatten()
                loss = criterion(output, target)
                loss.backward()
                # Record loss
                losses += loss.item()
                weight += len(data)
                optimizer.step()
        return model, losses / E / weight, weight / E

    def _send(self, state):
        """Duplicate the central model to the client"""
        model = self.Model().to(self.device)
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                parameter.data = state[name].clone()
        return model

    def global_update(self, state, lr, E=1):
        """Execute one round of serial global update"""
        parameters = []
        weights = []
        losses = 0

        for i in range(self.client_count):
            model, loss, data_count = self._client_update(i, self._send(state), lr, E)
            parameters.append(model.state_dict().copy())
            weights.append(data_count)
            losses += loss

        return self._fed_avg(parameters, weights), losses / self.client_count

    def federated_data(self, dataset):
        """Construct the federated dataset"""
        if self.client_count > len(dataset):
            return "ERROR"
        self.clients_data = [[] for _ in range(self.client_count)]
        for i, (data, target) in enumerate(dataset):
            self.clients_data[i % self.client_count].append((data.to(self.device), target.to(self.device)))


class ParallelFL(FederatedLearning):
    def __init__(self, Model, device, client_count, manager):
        super(ParallelFL, self).__init__(Model, device, client_count)
        self.Model = Model
        self.device = device
        self.client_count = client_count
        self.queue = manager.Queue()
        self.models=[Model().to(self.device) for i in range(self.client_count)]

    def _fed_avg(self, weights):
        """Execute FedAvg algorithm"""
        with torch.no_grad():
            clients_updates=[model.state_dict() for model in self.models]
            new_parameters = clients_updates[0].copy()
            weights = np.array(weights) / sum(weights)
            for name in new_parameters:
                new_parameters[name] = torch.zeros(new_parameters[name].shape).to(self.device)
            for idx, parameter in enumerate(clients_updates):
                for name in new_parameters:
                    new_parameters[name] += parameter[name] * weights[idx]
        return new_parameters.copy()

    def _client_update(self, client_id, lr, E):
        """Update the model on client"""
        model=self.models[client_id]
        optimizer = optim.SGD(model.parameters(), lr=lr)
        criterion = nn.MSELoss(reduction="sum")
        weight = 0
        losses = 0
        # print(client_id,self.clients_data[client_id])
        for e in range(E):
            for data, target in self.clients_data[client_id]:
                optimizer.zero_grad()
                output = model(data).flatten()
                loss = criterion(output, target)
                loss.backward()
                # Record loss
                losses += loss.item()
                weight += len(data)
                optimizer.step()
        with torch.no_grad():
            self.queue.put((losses / E / weight, weight / E))

    def _send(self, state):
        """Duplicate the central model to the client"""
        for model in self.models:
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    parameter.data = state[name].clone()

    def global_update(self, state, lr, E=1):
        """Execute one round of serial global update"""
        weights = []
        losses = 0
        # mp.spawn(self._client_update, args=(1, lr, E,), nprocs=self.client_count)
        self._send(state)
        mp.spawn(self._client_update, ( lr, E), nprocs=self.client_count)
        # print("end")
        for i in range(self.client_count):
            (loss, data_count) = self.queue.get()
            # parameters.append(model.to(self.device).state_dict().copy())
            weights.append(data_count)
            losses += loss
        # print(parameters)

        return self._fed_avg(weights), losses / self.client_count

    def federated_data(self, dataset):
        """Construct the federated dataset"""
        if self.client_count > len(dataset):
            return "ERROR"
        self.clients_data = [[] for _ in range(self.client_count)]
        for i, (data, target) in enumerate(dataset):
            self.clients_data[i % self.client_count].append((data.to(self.device), target.to(self.device)))
