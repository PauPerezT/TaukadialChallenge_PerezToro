#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Load MRI Videos

->data.py

Created on Thu Jan 18 18:08:00 2024 for the Taukadial Challenge

@author: P.A. Perez-Toro
@email:paula.andrea.perez@fau.de
"""
from VAD import eVAD, duration_feats
import torch
from torch.utils import data
from torch.utils.data import Dataset
import os
import numpy as np 
import pandas as pd
import av
from sklearn.utils import class_weight
import random
import json
import numpy as np
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

# Ignore warnings


from moviepy.editor import *
import librosa

from natsort import natsorted

from transformers import  AutoFeatureExtractor
#%%


class Dataset_Gen(Dataset):

    def __init__(self, files,  labels, lg, NR = True, balance = False):
        self.files = files
        self.audio_processor_en = AutoFeatureExtractor.from_pretrained("openai/whisper-large") #jonatasgrosman/whisper-large-zh-cv11
        self.audio_processor_zh = AutoFeatureExtractor.from_pretrained("jonatasgrosman/whisper-large-zh-cv11") #jonatasgrosman/whisper-large-zh-cv11
        self.audios = []
        self.labels = []
        self.texts = []
        self.fs =16000
        self.lg = lg
        self.audios_org = []
        #labels = self.label_extraction(labels)
        len_audio = []
        #Preprocessing
        for file,  label in zip(files,  labels):
            audio,_ = self.read_audio(file)


            self.audios.append(audio)
            self.audios_org.append(audio)
            self.labels.append(label)
            len_audio.append(len(audio))
        
        max_pad = np.max(np.hstack(len_audio))
        print('Max Lenght', max_pad)


        #Next code Norm
        for i,aud in enumerate(self.audios):
            if len(aud)<max_pad:
                pad_num = max_pad-len(aud)
                self.audios[i] = np.hstack((aud,np.zeros(pad_num)))
                

        self.labels = np.hstack(self.labels)
        self.inds = np.arange(0,len(self.labels))
        #self.audios = np.vstack(self.audios)
        if balance:
            inds_0 = np.where(self.labels == 0)[0]
            inds_1 = np.where(self.labels == 1)[0]

            if len(inds_0)>len(inds_1):
                random.shuffle(inds_0)
                inds_0 = inds_0[:len(inds_1)]
                self.inds = np.hstack((inds_0, inds_1))
                #self.audios = self.audios[inds]
                self.labels = self.labels[self.inds]
                self.lg = self.lg[self.inds]
            elif len(inds_0)<len(inds_1):
                random.shuffle(inds_1)
                inds_1 = inds_1[:len(inds_0)]
                self.inds = np.hstack((inds_0, inds_1))
                #self.audios = self.audios[inds]
                self.labels = self.labels[self.inds]
                self.lg = self.lg[self.inds]



    def __getitem__(self, index):
        
        if torch.is_tensor(index):
            index = index.tolist()

        #text =self.texts[index]
        audio = self.audios[index]
        audio_org = self.audios_org[index]
        lg = self.lg[index]


        if lg == 'zh':
            audio_tensor = self.audio_processor_zh(audio, sampling_rate=16000,
                                    return_tensors="pt",
                                    truncation=True,
                                    ppadding='longest').input_features[0]
                                        #, padding='max_length', truncation =True, max_length=3000

        else:
            audio_tensor = self.audio_processor_en(audio, sampling_rate=16000,
                                    return_tensors="pt",
                                    truncation=True,
                                    padding='longest').input_features[0]
                                        #, padding='max_length', truncation =True, max_length=3000


        #Extract timing
        #x = eVAD(audio, self.fs )

        feats = np.hstack(duration_feats(audio_org, self.fs )) #duration_feats(sig,fs = 16000, win=0.025,step=0.01)
 
        #label = text
        feats = torch.tensor(feats).type(torch.FloatTensor)

        label = torch.tensor(self.labels[index]).type(torch.LongTensor)


        return {"input_ids": audio_tensor, "Timing":feats, "labels": label, "language": lg}


    def __len__(self):
        

        return len(self.labels)

    def read_audio(self, path):

        audio, sr = librosa.load(path)
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        # audio.write_audiofile(sys.argv[2])  # 4.#
        return audio, 16000
    def class_weight(self):

        self.classWeights = torch.Tensor(
            class_weight.compute_class_weight('balanced', classes=np.unique(self.labels), y=self.labels))

        return self.classWeights




def get_dataset(path_set, path,  balance = True, task=1):
    #print('%-----Train tensors-----%')

    #path_scr, spk, 'videos'
    data = pd.read_csv(path_set)
    if task == 1:
        task='-1.wav'
        inds = np.hstack([np.hstack([i for i,file in enumerate(data['tkdname']) if task in file])])
    elif task == 2:
        task='-2.wav'
        inds = np.hstack([np.hstack([i for i,file in enumerate(data['tkdname']) if task in file])])
    elif task == 3:
        task='-3.wav'
        inds = np.hstack([np.hstack([i for i,file in enumerate(data['tkdname']) if task in file])])
    else:
        inds = np.hstack([np.hstack([i for i,file in enumerate(data['tkdname'])])])
    #print(data['file'],data['speaker'])
    files = np.hstack([os.path.join(path, file[:-4])+'.wav' for file in data['tkdname']])
    labels = np.hstack(np.array(data['dx']))
    
    lg = np.hstack([np.hstack([l[:2] for l in np.hstack(np.array(data['language'])) ])])
    labels_tmp = []
    for lb in labels:
        if lb == 'NC':
            labels_tmp.append(0)
        else:
            labels_tmp.append(1)
    labels = np.hstack(labels_tmp)


    class_weights = []
    Data = Dataset_Gen(files[inds], labels =labels[inds], lg = lg[inds],  balance = balance)
    #weigths=Data.class_weight()
    class_weights = Data.class_weight()

    return Data, class_weights


