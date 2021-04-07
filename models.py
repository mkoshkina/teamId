#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 10:03:19 2020

@author: maria
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

####################### Embdeding Network ##############################################        
class ContrastNN(nn.Module):
    def __init__(self):
        super(ContrastNN, self).__init__()
        ## encoder layers ##
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)  
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.l1 = nn.Linear(8192, 2048)
        self.l2 = nn.Linear(2048,1024)
        self.leakyReLU = nn.LeakyReLU(0.1)

    def encode(self, x):
        ## encode ##
        # add hidden layers with relu activation function
        # and maxpooling after
        x = self.leakyReLU(self.conv1(x))

        x = self.pool(x)

        # add second hidden layer
        x = self.leakyReLU(self.conv2(x))

        x = self.pool(x)  # compressed representation

        x = self.leakyReLU(self.conv3(x))

        x = self.pool(x) 

        x = x.reshape(x.size(0),-1)
        x = self.leakyReLU(self.l1(x))
        x = self.leakyReLU(self.l2(x))
        return x

    def forward(self, x):
        x = self.encode(x)
        return x
    
####################### Referee Classifier ##############################################        

class ConvClassifier(nn.Module):
    def __init__(self):
        super(ConvClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)  
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.l1 = nn.Linear(8192, 2048)
        self.l2 = nn.Linear(2048, 1)
        self.leakyReLU = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.leakyReLU(self.conv1(x))
        x = self.pool(x)
        x = self.leakyReLU(self.conv2(x))
        x = self.pool(x)  
        x = self.leakyReLU(self.conv3(x))
        x = self.pool(x) 
        x = x.reshape(x.size(0),-1)
        x = self.leakyReLU(self.l1(x))       
        x = F.sigmoid(self.l2(x))
        
        return x
    
    # specify threshold = -1 to get the value returned from a network or
    # specify a threshold between 0 and 1 to get a binary value
    def predict(self,x, threshold = 0.5):
        #Apply softmax to output. 
        pred = self.forward(x)
        ans = []
        #Pick the class based on threshold or return predicted value
        for t in pred:
            if threshold == -1:
                ans.append(t[0])
            elif t[0]<= threshold:
                ans.append(0)
            else:
                ans.append(1)
        return torch.tensor(ans)
    
############################ Autoencoder #################################################
        
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        ## encoder layers ##
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)  
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2, return_indices=True)
        self.l1 = nn.Linear(8192, 2048)
        self.l2 = nn.Linear(2048, 1024)

        
        ## decoder layers ##
        ## a kernel of 2 and a stride of 2 will increase the spatial dims by 2
        self.rl2 = nn.Linear(1024, 2048)
        self.rl1 = nn.Linear(2048, 8192)
        self.t_conv1 = nn.ConvTranspose2d(64, 32, 3, padding=1)
        self.t_conv2 = nn.ConvTranspose2d(32, 16, 3, padding=1)
        self.t_conv3 = nn.ConvTranspose2d(16, 3, 3, padding=1)
        self.unpool = nn.MaxUnpool2d(2, 2)

    def encode(self, x):
        ## encode ##
        # add hidden layers with relu activation function
        # and maxpooling after
        x = F.relu(self.conv1(x))

        x, self.ind1 = self.pool(x)

        # add second hidden layer
        x = F.relu(self.conv2(x))

        x, self.ind2 = self.pool(x)  # compressed representation

        x = F.relu(self.conv3(x))

        x, self.ind3 = self.pool(x) 

        x = x.reshape(x.size(0),-1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        
        return x
    
    def decode(self, x):
        ## decode ##
        x = F.relu(self.rl2(x))
        x = F.relu(self.rl1(x))
        x = x.reshape(x.size(0),64, 16, 8)
        
        x = self.unpool(x, self.ind3)

        # add transpose conv layers, with relu activation function
        x = F.relu(self.t_conv1(x))

        x = self.unpool(x, self.ind2)
        
        
        x = F.relu(self.t_conv2(x))

        x = self.unpool(x, self.ind1)
        # output layer (with sigmoid for scaling from 0 to 1)
        x = F.sigmoid(self.t_conv3(x))   
        
        
        return x

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x  
