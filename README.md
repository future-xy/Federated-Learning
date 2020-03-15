# Federated Learning
 **A SIMULATION framework of Federated Learning based on PyTorch**

### Introduction

This repository contains a **simulation** framework of Federated Learning implemented by PyTorch. This framework is an extraction of my recent work on Federated Learning (FL). 

I combined FL with Linear Regression, Deep Neural Network, and Convolutional Neural Network in the past few months. And I also tried to add Differential Privacy (DP) to FL. Every time when I built a new model, I had to copy the last code and then did some modification on it. This is absolutely inefficient and hard to develop advanced features. Therefore, I extract this framework to simplify future development. 

Note that this framework, so far, is only designed for scientific simulation about FL. In the future, a practical FL framework may be implemented. 

### Multi-process

To accelerate the simulation, a multi-process version FL is implemented. Due to the enormous time cost to create and destroy subprocesses, this model should only be used when training time is much more than the spawn cost and the number of client is not too much. 

The interfaces are unified in both serial and parallel FL model. Therefore, you may test both models for a few epochs and decide which is suitable in your scenario.

### Dataset

An efficient federated dataset has been built in FLsim.federated_data. Generally, it's no need to know how it works. 

To construct the federated data, you need to pass a dataset whose type is *torch.utils.data.Dataset*, i.e., your custom dataset inherited from *Dataset*.

~~~python
from torch.utils.data import TensorDataset
train_data = TensorDataset(X, y)
~~~

Besides, you need to state the data owner for each data and the client id should in the range of $[0,\,client count)$, like below.

~~~
clients = [i % client_count for i in range(len(train_data))]
~~~

This function will partition the data to clients as claimed in *clients*

~~~
FL = SerialFL(Model, device, client_count)
FL.federated_data(train_data, clients, batch_size)
~~~

### Implemented

- Federated dataset 
- FedAvg (Single and multi process)
- Federated Linear Regression model on Boston dataset

### Ongoing

- Combination with DP
- Combination with Secure MPC
- Test more Federated model (YOLOv3 ...)

##### Any suggestions or bug reports are welcome!

##### Appreciate for your STAR! 😘