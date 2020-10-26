#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 00:02:30 2020

@author: krishna
"""

import torch
import torch.nn as nn
from model.transformers import TransformerEncoderLayer


class ConvBlocks(nn.Module):
    def __init__(self,):
        super(ConvBlocks, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1),padding=(1,1)),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1),padding=(1,1)),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1),padding=(1,1)),
            nn.BatchNorm2d(128),
            nn.Conv2d(128,128, kernel_size=(3, 3), stride=(1, 1),padding=(1,1)),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1,1)),
            nn.BatchNorm2d(256),
            nn.Conv2d(256,256, kernel_size=(3, 3), stride=(1, 1), padding=(1,1)),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1,1)),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
            nn.ReLU(),
            
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1,1)),
            nn.BatchNorm2d(512),
            nn.Conv2d(512,512, kernel_size=(3, 3), stride=(1, 1), padding=(1,1)),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1,1)),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(kernel_size=(2,1), stride=(2,1)),
            nn.ReLU(),
            
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1),padding=(1,1)),
            nn.BatchNorm2d(512),
            nn.Conv2d(512,512, kernel_size=(3, 3), stride=(1, 1),padding=(1,1)),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1),padding=(1,1)),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(kernel_size=(2,1), stride=(2,1)),
            nn.ReLU(),
        )
        
    def forward(self,inputs):
       out = self.conv(inputs)
       out = out.flatten(start_dim=1, end_dim=2)
       return out
   


class Model(nn.Module):
    def __init__(self,num_classes=9):
        super(Model, self).__init__()
        self.num_classes = num_classes
        
        self.conv = ConvBlocks()
        self.blstm = nn.LSTM(1024, hidden_size=int(320/2),bidirectional=True, batch_first=True)
        self.mha = TransformerEncoderLayer(embed_dim=320, num_heads=10,temp=0.2)
        self.fc1 = nn.Linear(320, 512)
        self.fc2 = nn.Linear(512, self.num_classes)
    
    def forward(self, inputs):
        cnn_out = self.conv(inputs)
        cnn_out = cnn_out.permute(0,2,1)
        rnn_out,_ = self.blstm(cnn_out)
        rnn_out = rnn_out.permute(1,0,2)
        mha_out = self.mha(rnn_out)
        mha_out = mha_out.permute(1,0,2)
        pooled = torch.mean(mha_out, dim=1)
        fc1_out = self.fc1(pooled)
        out = self.fc2(fc1_out)
        return out
        
        
        
        
        
        
        
        