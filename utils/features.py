#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 00:55:25 2020

@author: krishna
"""

import librosa
import numpy as np

def load_wav(audio_filepath, sr, max_duration_sec=10.0):
    audio_data,fs  = librosa.load(audio_filepath,sr=16000)
    win_length = int(fs*max_duration_sec)
    if len(audio_data) < win_length:
        diff = win_length-len(audio_data)
        create_arr = np.zeros([1,diff])
        final_data  = np.concatenate((audio_data,create_arr[0]))
        audio_data = final_data
        ret_data = audio_data
    else:
        ret_data = audio_data[:win_length]
    return ret_data
    


def mel_spec_from_wav(wav, hop_length, win_length, n_mels=64):
    #linear = librosa.stft(wav, n_fft=n_fft, win_length=win_length, hop_length=hop_length)
    mel_feats=librosa.feature.melspectrogram(wav, sr=16000, n_mels=n_mels, win_length=256, hop_length=128)
    
    return mel_feats.T

def load_data(path, win_length=256, sr=16000, hop_length=128,n_mels=64, spec_len=1250):
    wav = load_wav(path, sr=sr)
    linear_spect = mel_spec_from_wav(wav, hop_length, win_length, n_mels=64)
    mag, _ = librosa.magphase(linear_spect)  # magnitude
    mag_T = mag.T
    freq, time = mag_T.shape
    
    spec_mag = mag_T[:, :spec_len]
    # preprocessing, subtract mean, divided by time-wise var
    mu = np.mean(spec_mag, 0, keepdims=True)
    std = np.std(spec_mag, 0, keepdims=True)
    ret_spec=(spec_mag - mu) / (std + 1e-5)
    
    return ret_spec
