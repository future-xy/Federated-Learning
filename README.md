# Federated Learning
 **A SIMULATION framework of Federated Learning based on PyTorch**

### Introduction

This repository contains a **simulation** framework of Federated Learning implemented by PyTorch. This framework is an extraction of my recent work on Federated Learning (FL). 

I combined FL with Linear Regression, Deep Neural Network, and Convolutional Neural Network in the past few months. And I also tried to add Differential Privacy (DP) to FL. Every time when I built a new model, I had to copy the last code and then did some modification on it. This is absolutely inefficient and hard to develop advanced features. Therefore, I extract this framework to simplify future development. 

Note that this framework, so far, is only designed for scientific simulation about FL. In the future, a practical FL framework may be implemented. 

### Implemented

- IID data partition. 
- FedAvg (Single process)
- Test Federated Linear Regression model on Boston dataset

### Ongoing

- Multi-process FedAvg
- Combination with DP
- Combination with Secure MPC
- Test more Federated model (YOLOv3 ...)

##### Any suggestions or bug reports are welcome!