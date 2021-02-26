import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp
import numpy as np
import os
from os import path
from csv import reader
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import re

class Dataset(Dataset):
    def __init__(self, root_dir):
        self.input = root_dir + "/X.txt"
        self.output = root_dir + "/y.txt"
        self.input_np = None
        self.output_np = None

        with open(self.output, 'r') as f:
            i = 0;
            for line in f.readlines():
                if(i == 0):
                    #self.output_np = np.array([float(digit) for digit in line.split(',')])
                    if(float(line) == 0):
                        #self.output_np = [np.array([1, 0])]
                        self.output_np = np.array([0])
                    else:
                        #self.output_np = [np.array([0, 1])]
                        self.output_np = np.array([1])
                    i = 1
                else:
                    if(float(line) == 0):
                        #self.output_np = np.append(self.output_np, np.array([1, 0]))
                        self.output_np = np.append(self.output_np, 0)
                    else:
                        #self.output_np = np.append(self.output_np, np.array([0, 1]))
                        self.output_np = np.append(self.output_np, 1)

                    
                    #self.output_np = np.append(self.output_np, [np.array([float(digit) for digit in line.split(',')])])


        #self.output_np = np.reshape(self.output_np, (-1, 1))
        #print(self.output_np.shape)
        #print(self.output_np[0, :])
        #print(self.output_np[1, :])

        with open(self.input, 'r') as f:
            i = 0
            for line in f.readlines():
                if(i == 0):
                    #self.input_np = np.array([[]])
                    self.input_np = [np.array([float(digit) for digit in line.split(',')])]
                    i = 1
                else:
                    self.input_np = np.append(self.input_np, np.array([float(digit) for digit in line.split(',')]))

        self.input_np = np.reshape(self.input_np, (-1, 5))

        #print(self.input_np.shape)
        #print(self.input_np[0, :])
        #print(self.input_np[1, :])
        #exit()

        #print(type(self.input_np))
        #print(self.input_np.shape)
        #print(type(self.output_np))
        #print(self.output_np.shape)


    def __len__(self):
        return len(self.output_np)

    def __getitem__(self, idx):
        input1 = self.input_np[idx, :]
        target1 = self.output_np[idx]

        #assert(input1.shape[0] == 5)
        #assert(target1.shape[0] == 1)

        inputs = torch.from_numpy(input1)
        #targets = torch.from_numpy(target1)
        targets = target1
        inputs = inputs.type(torch.FloatTensor)
        #targets = targets.type(torch.FloatTensor)
        #targets = targets.type(torch.FloatTensor)

        sample = {'target': targets,
                  'input': inputs
                  }
        return sample

class my_net(nn.Module):
    def __init__(self):
        super(my_net, self).__init__()
        self.input = None
        self.num_features = 5

        self.b1 = nn.BatchNorm1d(self.num_features)
        self.fc1 = nn.Linear(5, 7)
        self.fc2 = nn.Linear(7, 10)
        self.fc3 = nn.Linear(10, 10)
        #self.fc4 = nn.Linear(10, 10)
        #self.fc5 = nn.Linear(10, 10)
        #self.fc6 = nn.Linear(10, 10)
        #self.fc7 = nn.Linear(10, 10)
        #self.fc8 = nn.Linear(10, 10)
        self.fc9 = nn.Linear(10, 5)
        self.fc10 = nn.Linear(5, 2)

    def forward(self, inputs):

        x = self.b1(inputs)
        x = self.fc1(x)
        x = F.leaky_relu(x)

        x = self.fc2(x)
        x = F.leaky_relu(x)

        x = self.fc3(x)
        x = F.leaky_relu(x)

        x = self.fc4(x)
        x = F.leaky_relu(x)

        x = self.fc5(x)
        x = F.leaky_relu(x)

        x = self.fc6(x)
        x = F.leaky_relu(x)

        x = self.fc7(x)
        x = F.leaky_relu(x)

        x = self.fc8(x)
        x = F.leaky_relu(x)

        x = self.fc9(x)
        x = F.leaky_relu(x)

        x = self.fc10(x)
        x = F.leaky_relu(x)

        #output = F.log_softmax(x, dim=1)
        output = F.softmax(x, dim=1)

        return output

if __name__ == '__main__':
    root = "./"
    batch = 2
    epochs = 25


    model = my_net()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"


    model.to(device)

    learn_rate = 0.0001;
    epsilon = 1e-8

    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate, eps=epsilon)

    decayRate = 0.8
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)

    model_file = "weight.pt"
    
    trainset = Dataset(root_dir="./train")
    valset = Dataset(root_dir="./val")
    testset = Dataset(root_dir="./test")

    train_loader = DataLoader(trainset, batch_size=batch, drop_last=False, shuffle=False)
    val_loader = DataLoader(valset, batch_size=batch, drop_last=False, shuffle=False)
    test_loader = DataLoader(testset, batch_size=batch, drop_last=False, shuffle=False)

    if (not(path.exists(model_file))):
        for k in range(epochs):
            print("Epoch: ", k)

            total = 0
            correct = 0

            for batch_index, batch_samples in enumerate(train_loader):
                targets, inputs = batch_samples['target'], batch_samples['input']
                targets = targets.to(device)
                inputs = inputs.to(device)

                #print(inputs.shape)

                outputs = model(inputs)

                _, predicted = torch.max(outputs.data, 1)


                total += targets.size(0)
                correct += (predicted == targets).sum().item()

                loss1 = loss(outputs, targets)

                model.zero_grad()
                loss1.backward()
                optimizer.step()



            print('Accuracy of the network epoch %d is %f %%' % (k, (100 * float(correct) / float(total))))

            #scheduler.step()

