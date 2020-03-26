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

    def _send(self, state):
        """Duplicate the central model to the client"""
        for model in self.models:
            model.load_state_dict(state.copy())
        self.weights = [0] * self.client_count
        self.losses = [0] * self.client_count

    def _fed_avg(self):
        """Execute FedAvg algorithm"""
        self.weights = np.array(self.weights) / sum(self.weights)
        with torch.no_grad():
            clients_updates = [model.state_dict() for model in self.models]
            new_parameters = clients_updates[0].copy()
            for name in new_parameters:
                new_parameters[name] = torch.zeros(new_parameters[name].shape).to(self.device)
            for client_id, parameter in enumerate(clients_updates):
                for name in new_parameters:
                    new_parameters[name] += parameter[name] * self.weights[client_id]
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
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                # Record loss
                losses += loss.item()
                weight += len(data)
                optimizer.step()
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
        self.queue = manager.Queue()

    # def _send(self, state):
    #     """Duplicate the central model to the client"""
    #     for model in self.models:
    #         with torch.no_grad():
    #             for name, parameter in model.named_parameters():
    #                 parameter.data = state[name].clone()
    #     self.weights = [0] * self.client_count
    #     self.losses = [0] * self.client_count

    def _recv(self):
        for _ in range(self.client_count):
            (client_id, weight, loss) = self.queue.get()
            self.weights[client_id] = weight
            self.losses[client_id] = loss

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
                data, target = data.to(self.device), target.to(self.device)
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


class FedSGD_LocalDP(FLBase):
    def __init__(self, Model, device, client_count, optimizer, criterion, DP_noise, clip=None):
        super(FedSGD_LocalDP, self).__init__(Model, device, client_count, optimizer, criterion)
        self.DP_noise = DP_noise
        self.clip = clip
        self.global_model = Model().to(device)

    def _client_update(self, client_id, lr, E):
        """Update the model on client"""
        model = self.models[client_id]
        criterion = self.criterion(reduction="sum")
        dataloader = self.client_dataloader[client_id]
        weight = 0
        losses = 0
        for data, target in dataloader:
            data, target = data.to(self.device), target.to(self.device)
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            # Record loss
            losses += loss.item()
            weight += len(data)
        self.weights[client_id] = weight / E
        self.losses[client_id] = losses / E / weight
        # Add Differential Privacy noise here
        with torch.no_grad():
            for para in model.parameters():
                if para.requires_grad:
                    if self.clip is not None:
                        para.grad /= max(1, torch.norm(para.grad, 2) / self.clip)
                    para.grad += torch.from_numpy(self.DP_noise(size=para.shape)).to(self.device)
            

    def _recv(self):
        self.weights = np.array(self.weights) / sum(self.weights)
        with torch.no_grad():
            grads = {}
            for name, para in self.global_model.named_parameters():
                grads[name] = torch.zeros(para.shape).to(self.device)
            for id, client in enumerate(self.models):
                for name, para in client.named_parameters():
                    grads[name] += para.grad * self.weights[id]
            for name, para in self.global_model.named_parameters():
                para.grad = grads[name]

    def global_update(self, state, lr, E):
        self._send(state)
        self.global_model.load_state_dict(state.copy())
        global_optimizer = self.optimizer(self.global_model.parameters(), lr=lr)
        global_optimizer.zero_grad()
        for i in range(self.client_count):
            self._client_update(i, lr, E)
        self._recv()
        global_optimizer.step()
        return self.global_model.state_dict(), sum(self.losses) / self.client_count
