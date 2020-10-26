#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 01:03:39 2020

@author: krishna
"""


import numpy as np
import torch
from utils import features
class SpeechDataGenerator():
    """Speech dataset."""

    def __init__(self, manifest):
        """
        Read the textfile and get the paths
        """
        self.audio_links = [line.rstrip('\n').split('\t')[0] for line in open(manifest)]
        self.labels = [line.rstrip('\n').split('\t')[1] for line in open(manifest)]
        
        
    def __len__(self):
        return len(self.audio_links)

    def __getitem__(self, idx):
        audio_link =self.audio_links[idx]
        label = int(self.labels[idx])
        specgram = features.load_data(audio_link)
        sample = {'spec': torch.from_numpy(np.ascontiguousarray(specgram)),'labels': torch.from_numpy(np.ascontiguousarray(label))}
        return sample
    
    

def collate_fn(batch):
   specs = []
   labels = []
   for sample in batch:
        specs.append(sample['spec'].unsqueeze(0))
        labels.append(sample['labels'][0])
   return specs, labels