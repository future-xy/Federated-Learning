#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Name    : federated_data.py
# Time    : 3/15/2020 3:34 PM
# Author  : Fu Yao
# Mail    : fy38607203@163.com

from torch.utils.data import Dataset


class FederatedDataset(Dataset):
    def __init__(self, dataset: Dataset, clients, client_id):
        """This dataset is only responsible for mapping data to each client"""
        super(FederatedDataset, self).__init__()
        self.dataset = dataset
        self.map = []
        self.id = client_id
        for i, id in enumerate(clients):
            if id == self.id:
                self.map.append(i)
        self.len = len(self.map)
        # You may add your custom dataset operations below

    def __getitem__(self, index):
        return self.dataset.__getitem__(self.map[index])

    def __len__(self):
        return self.len
