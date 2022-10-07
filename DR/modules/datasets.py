#%%
from turtle import down
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
    def __init__(self, config, downstream=False):
        foldername = 'pendulum_DR'
        self.name = ['light', 'angle', 'length', 'position', 'background', 'target']
        train_imgs = [x for x in os.listdir('./modules/causal_data/{}/train'.format(foldername)) if x.endswith('png')]
        
        """labeled ratio: semi-supervised learning"""
        train_imgs = train_imgs[:int(len(train_imgs) * config["labeled_ratio"])]
        
        train_x = []
        for i in tqdm.tqdm(range(len(train_imgs)), desc="train data loading"):
            train_x.append(np.array(
                Image.open("./modules/causal_data/{}/train/{}".format(foldername, train_imgs[i])).resize(
                    (config["image_size"], config["image_size"])))[:, :, :3])
        self.x_data = (np.array(train_x).astype(float) - 127.5) / 127.5
        
        label = np.array([x[:-4].split('_')[1:] for x in train_imgs]).astype(float)
        if not downstream:
            label[:, :4] = label[:, :4] - label[:, :4].mean(axis=0)
            self.std = label[:, :4].std(axis=0)
            """bounded label: normalize to (0, 1)"""
            if config["label_normalization"]: 
                label[:, :4] = (label[:, :4] - label[:, :4].min(axis=0)) / (label[:, :4].max(axis=0) - label[:, :4].min(axis=0))
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
        foldername = 'pendulum_DR'
        train_imgs = [x for x in os.listdir('./modules/causal_data/{}/train'.format(foldername)) if x.endswith('png')]
        
        train_x = []
        for i in tqdm.tqdm(range(len(train_imgs)), desc="train data loading"):
            train_x.append(np.array(
                Image.open("./modules/causal_data/{}/train/{}".format(foldername, train_imgs[i])).resize(
                    (config["image_size"], config["image_size"])))[:, :, :3])
        self.x_data = (np.array(train_x).astype(float) - 127.5) / 127.5
        
    def __len__(self): 
        return len(self.x_data)

    def __getitem__(self, idx): 
        x = torch.FloatTensor(self.x_data[idx])
        return x
#%%
class TestDataset(Dataset): 
    def __init__(self, config, downstream=False):
        foldername = 'pendulum_DR'
        self.name = ['light', 'angle', 'length', 'position', 'background', 'target']
        test_imgs = [x for x in os.listdir('./modules/causal_data/{}/test'.format(foldername)) if x.endswith('png')]
        
        test_x = []
        for i in tqdm.tqdm(range(len(test_imgs)), desc="test data loading"):
            test_x.append(np.array(
                Image.open("./modules/causal_data/{}/test/{}".format(foldername, test_imgs[i])).resize(
                    (config["image_size"], config["image_size"])))[:, :, :3])
        self.x_data = (np.array(test_x).astype(float) - 127.5) / 127.5
        
        label = np.array([x[:-4].split('_')[1:] for x in test_imgs]).astype(float)
        if not downstream:
            label[:, :4] = label[:, :4] - label[:, :4].mean(axis=0)
            self.std = label[:, :4].std(axis=0)
            """bounded label: normalize to (0, 1)"""
            if config["label_normalization"]: 
                label[:, :4] = (label[:, :4] - label[:, :4].min(axis=0)) / (label[:, :4].max(axis=0) - label[:, :4].min(axis=0))
        self.y_data = label

    def __len__(self): 
        return len(self.x_data)

    def __getitem__(self, idx): 
        x = torch.FloatTensor(self.x_data[idx])
        y = torch.FloatTensor(self.y_data[idx])
        return x, y
#%%