#%%
import tqdm
import os
import numpy as np
import pandas as pd

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import Dataset
#%%
class TabularDataset(Dataset): 
    def __init__(self, config):
        """
        load dataset: Personal Loan
        Reference: https://www.kaggle.com/datasets/teertha/personal-loan-modeling
        """
        df = pd.read_csv('./data/Bank_Personal_Loan_Modelling.csv')
        df = df.sample(frac=1, random_state=1).reset_index(drop=True)
        df = df.drop(columns=['ID'])
            
        self.continuous = ['CCAvg', 'Mortgage', 'Income', 'Experience', 'Age']
        self.topology = [['CCAvg', 'Mortgage'], ['Experience', 'Age'], ['Income']]
        self.flatten_topology =  [self.continuous.index(item) for sublist in self.topology for item in sublist]
            
        df = df[self.continuous]
        
        train = df.iloc[:4000]
        # continuous
        mean = train.mean(axis=0)
        std = train.std(axis=0)
        train = (train - mean) / std # local statistic
        df = (df - mean) / std
        
        from sklearn.decomposition import PCA
        pca1 = PCA(n_components=1).fit_transform(df[self.topology[0]])
        pca2 = PCA(n_components=1).fit_transform(df[self.topology[1]])
        pca3 = PCA(n_components=1).fit_transform(df[self.topology[2]])
        label = np.concatenate([pca1, pca2, pca3], axis=1)
        
        """bounded label: normalize to (0, 1)"""
        if config["label_normalization"]: 
            # label = (label - label.min(axis=0)) / (label.max(axis=0) - label.min(axis=0)) # global statistic
            label = (label - label.mean(axis=0)) / label.std(axis=0)
        self.label = label[:4000, :]
        
        self.train = train
        self.x_data = train.to_numpy()
        
    def __len__(self): 
        return len(self.x_data)

    def __getitem__(self, idx): 
        x = torch.FloatTensor(self.x_data[idx])
        y = torch.FloatTensor(self.label[idx])
        return x, y
#%%
class TestTabularDataset(Dataset): 
    def __init__(self, config):
        """
        load dataset: Personal Loan
        Reference: https://www.kaggle.com/datasets/teertha/personal-loan-modeling
        """
        df = pd.read_csv('./data/Bank_Personal_Loan_Modelling.csv')
        df = df.sample(frac=1, random_state=1).reset_index(drop=True)
        df = df.drop(columns=['ID'])
            
        self.continuous = ['CCAvg', 'Mortgage', 'Income', 'Experience', 'Age']
        self.topology = [['CCAvg', 'Mortgage'], ['Experience', 'Age'], ['Income']]
        self.flatten_topology =  [self.continuous.index(item) for sublist in self.topology for item in sublist]
        
        df = df[self.continuous]
        
        train = df.iloc[:4000]
        test = df.iloc[4000:]
        # continuous
        mean = train.mean(axis=0)
        std = train.std(axis=0)
        test = (test - mean) / std # local statistic
        df = (df - mean) / std
        
        from sklearn.decomposition import PCA
        pca1 = PCA(n_components=1).fit_transform(df[self.topology[0]])
        pca2 = PCA(n_components=1).fit_transform(df[self.topology[1]])
        pca3 = PCA(n_components=1).fit_transform(df[self.topology[2]])
        label = np.concatenate([pca1, pca2, pca3], axis=1)
        
        """bounded label: normalize to (0, 1)"""
        if config["label_normalization"]: 
            # label = (label - label.min(axis=0)) / (label.max(axis=0) - label.min(axis=0)) # global statistic
            label = (label - label.mean(axis=0)) / label.std(axis=0)
        self.label = label[4000:, :]
        
        self.test = test
        self.x_data = test.to_numpy()
        
    def __len__(self): 
        return len(self.x_data)

    def __getitem__(self, idx): 
        x = torch.FloatTensor(self.x_data[idx])
        y = torch.FloatTensor(self.label[idx])
        return x, y
#%%