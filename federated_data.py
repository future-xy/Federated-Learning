#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Name    : federated_data.py
# Time    : 3/5/2020 6:50 PM
# Author  : Fu Yao
# Mail    : fy38607203@163.com

def construct_federated_dataset(dataloader, client_count, mode="iid"):
    if client_count > len(dataloader):
        return "ERROR"
    clients_data = [[] for _ in range(client_count)]
    for i, (data, target) in enumerate(dataloader):
        clients_data[i % client_count].append((data, target))
    return clients_data
