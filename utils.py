#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Name    : utils.py
# Time    : 3/5/2020 7:03 PM
# Author  : Fu Yao
# Mail    : fy38607203@163.com

import torch


def eval(device, model, datas, criterion):
    """Eval the model"""
    losses = 0
    model.eval()
    with torch.no_grad():
        for data, target in datas:
            output = model(data.to(device)).flatten()
            losses += criterion(output.flatten(), target.to(device)).item()
    return losses / len(datas.dataset)
