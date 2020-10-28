#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 01:09:07 2020

@author: krishna

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
     http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


"""


import torch.nn as nn
import argparse
import numpy as np
from torch import optim
from model.model import Model
from DataGenerator import SpeechDataGenerator, collate_fn
from torch.utils.data import DataLoader
import torch
from sklearn.metrics import classification_report
torch.multiprocessing.set_sharing_strategy('file_system')



def arg_parser():
    ########## Argument parser
    parser = argparse.ArgumentParser(add_help=False)
    
    parser.add_argument('-training_filepath',type=str,default='meta/train.txt')
    parser.add_argument('-testing_filepath',type=str, default='meta/eval.txt')
    parser.add_argument('-num_classes', action="store_true", default=9)
    parser.add_argument('-batch_size', action="store_true", default=10)
    parser.add_argument('-use_gpu', action="store_true", default=True)
    parser.add_argument('-num_epochs', action="store_true", default=100)
    parser.add_argument('-temp', action="store_true", default=0.2)
    
    return parser.parse_args()


def train(model, data_loader, device, optimizer, criterion, epoch):
    model.train()
    total_loss =[]
    gt_labels = []
    pred_labels =[]
    for i_batch, sample_batched in enumerate(data_loader):
        features = torch.stack(sample_batched[0])
        labels = torch.stack(sample_batched[1])
        features, labels = features.to(device), labels.to(device)
        optimizer.zero_grad()
        pred = model(features)
        loss = criterion(pred, labels)
        loss.backward()
        optimizer.step()
        total_loss.append(loss.item())
        gt_labels = gt_labels + list(labels.detach().cpu().numpy())
        pred_labels = pred_labels + list(np.argmax(pred.detach().cpu().numpy(),axis=1))
        
        
    mean_loss = np.mean(np.asarray(total_loss))
    print(f'Training loss {mean_loss} after {epoch} epochs') 
    
    target_names = ['social_activity','dishwashing','watching_tv', 'absence', 'vacuum_cleaner',
                    'other','working','eating','cooking']
    print(classification_report(gt_labels, pred_labels, target_names=target_names, digits=4))
        
    
def evaluation(model, data_loader, device, optimizer, criterion, epoch):
    model.eval()
    total_loss =[]
    gt_labels = []
    pred_labels =[]
    with torch.no_grad():
        for i_batch, sample_batched in enumerate(data_loader):
            features = torch.stack(sample_batched[0])
            labels = torch.stack(sample_batched[1])
            features, labels = features.to(device), labels.to(device)
            pred = model(features)
            loss = criterion(pred, labels)
            total_loss.append(loss.item())
            gt_labels = gt_labels + list(labels.detach().cpu().numpy())
            pred_labels = pred_labels +list( np.argmax(pred.detach().cpu().numpy(),axis=1))
            
    mean_loss = np.mean(np.asarray(total_loss))
    print(f'Testing loss {mean_loss} after {epoch} epochs') 
    
    target_names = ['social_activity','dishwashing','watching_tv', 'absence', 'vacuum_cleaner',
                    'other','working','eating','cooking']
    
    print(classification_report(gt_labels, pred_labels, target_names=target_names, digits=4))
        
    
    


def main():
    args = arg_parser()
    ### Data loaders
    dataset_train = SpeechDataGenerator(manifest=args.training_filepath)
    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True,collate_fn=collate_fn) 
    
    dataset_eval = SpeechDataGenerator(manifest=args.testing_filepath)
    dataloader_eval = DataLoader(dataset_eval, batch_size=args.batch_size,collate_fn=collate_fn)
    ## Model related
    
    if args.use_gpu:
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
    else:
        device='cpu'
    
    model = Model(num_classes=args.num_classes)
    model = model.to(device) 
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-05, betas=(0.9, 0.98), eps=1e-9)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(1,args.num_epochs+1):
        train(model, dataloader_train, device, optimizer, criterion, epoch)
        evaluation(model, dataloader_eval, device, optimizer, criterion, epoch)
    
if __name__=='__main__':
    main()
