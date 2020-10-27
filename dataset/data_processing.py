#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 22:49:32 2020

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

import os
import glob
import json
from typing import List

class DCase2018:
        
    def __init__(self, dataset_root: str):
        self.dataset_folder = dataset_root
        
    def _find_folder(self,):
        self.audio_folders = glob.glob(self.dataset_folder+'/*.audio*')
        label_folders = glob.glob(self.dataset_folder+'/*.meta')
        if label_folders:
            self.label_folder = label_folders[0]
    
    def _find_audios(self,) -> List[str]:
        audio_files = [glob.glob(audio_folder+'/DCASE2018-task5-dev/audio/*.wav') for audio_folder in self.audio_folders]
        audios = sum(audio_files,[])
        return audios
    
    def _get_label_ids(self,):    
        meta_file = os.path.join(self.label_folder,'DCASE2018-task5-dev/meta.txt')
        if not os.path.exists(meta_file):
            print(f'Please check if your download folders contains meta folder')
        label_dict =dict()
        read_data = [line.rstrip('\n') for line in open(meta_file)]
        for row in read_data:
            filename = row.split('\t')[0].split('/')[-1]
            label_dict[filename] = row.split('\t')[1]
        
        labels = set(label_dict.values())
        return labels, label_dict
        
    def _create_manifest(self, mode='train'):
        
        self._find_folder()
        audio_files = self._find_audios()
        labels, labels_dict = self._get_label_ids()
        print(f'Creating label dictionary at meta/labels.json')
        label_ids=dict()
        with open('meta/labels.json','w') as fid:
            for i in range(len(list(labels))):
                json.dump({list(labels)[i]:i}, fid)
                fid.write('\n')
                label_ids[list(labels)[i]]=i
        save_file = 'meta/'+mode+'.txt'
        with open(save_file,'w') as fid:    
            for audio_filepath in audio_files:
                to_write = audio_filepath+'\t'+str(label_ids[labels_dict[audio_filepath.split('/')[-1]]])
                fid.write(to_write+'\n')
        print(f'created {mode} set')


if __name__=='__name__':
    
    train_folder = '/media/newhd/DCASE_2018/Dev'
    dataset = DCase2018(train_folder)
    dataset._create_manifest(mode='train')
    
    eval_folder = '/media/newhd/DCASE_2018/Eval'
    dataset._create_manifest(mode='eval')
    
    
    
    
