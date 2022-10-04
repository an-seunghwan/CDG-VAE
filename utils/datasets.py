#%%
import tqdm
from PIL import Image
import os
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import Dataset
#%%
class LabeledDataset(Dataset): 
    def __init__(self, config):
        if config["DR"]:
            foldername = 'pendulum_DR'
            self.name = ['light', 'angle', 'length', 'position', 'background', 'target']
        else:
            foldername = 'pendulum_real'
            self.name = ['light', 'angle', 'length', 'position', 'target']
        train_imgs = [x for x in os.listdir('./utils/causal_data/{}/train'.format(foldername)) if x.endswith('png')]
        
        """labeled ratio"""
        train_imgs = train_imgs[:int(len(train_imgs) * config["labeled_ratio"])]
        
        train_x = []
        for i in tqdm.tqdm(range(len(train_imgs)), desc="train data loading"):
            train_x.append(np.array(
                Image.open("./utils/causal_data/{}/train/{}".format(foldername, train_imgs[i])).resize(
                    (config["image_size"], config["image_size"])))[:, :, :3])
        self.x_data = (np.array(train_x).astype(float) - 127.5) / 127.5
        
        label = np.array([x[:-4].split('_')[1:] for x in train_imgs]).astype(float)
        label = label - label.mean(axis=0)
        self.std = label.std(axis=0)
        """bounded label: normalize to (0, 1)"""
        if config["label_normalization"]: 
            label = (label - label.min(axis=0)) / (label.max(axis=0) - label.min(axis=0))
        self.y_data = label

    def __len__(self): 
        return len(self.x_data)

    def __getitem__(self, idx): 
        x = torch.FloatTensor(self.x_data[idx])
        y = torch.FloatTensor(self.y_data[idx])
        return x, y
#%%
class UnLabeledDataset(Dataset): 
    def __init__(self, config):
        if config["DR"]:
            foldername = 'pendulum_DR'
        else:
            foldername = 'pendulum_real'
        train_imgs = [x for x in os.listdir('./utils/causal_data/{}/train'.format(foldername)) if x.endswith('png')]
        
        train_x = []
        for i in tqdm.tqdm(range(len(train_imgs)), desc="train data loading"):
            train_x.append(np.array(
                Image.open("./utils/causal_data/{}/train/{}".format(foldername, train_imgs[i])).resize(
                    (config["image_size"], config["image_size"])))[:, :, :3])
        self.x_data = (np.array(train_x).astype(float) - 127.5) / 127.5
        
    def __len__(self): 
        return len(self.x_data)

    def __getitem__(self, idx): 
        x = torch.FloatTensor(self.x_data[idx])
        return x
#%%