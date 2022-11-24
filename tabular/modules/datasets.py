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
def interleave_float(a: float, b: float):
    a_rest = a
    b_rest = b
    result = 0
    dst_pos = 1.0  # position of written digit
    while a_rest != 0 or b_rest != 0:
        dst_pos /= 10  # move decimal point of write
        a_rest *= 10  # move decimal point of read
        result += dst_pos * (a_rest // 1)
        a_rest %= 1  # remove current digit

        dst_pos /= 10
        b_rest *= 10
        result += dst_pos * (b_rest // 1)
        b_rest %= 1
    return result
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
        self.topology = [['Mortgage', 'Income'], ['Experience', 'Age'], ['CCAvg']]
        self.flatten_topology =  [self.continuous.index(item) for sublist in self.topology for item in sublist]
        
        df = df[self.continuous]
        
        min_ = df.min(axis=0)
        max_ = df.max(axis=0)
        train = df.iloc[:4000]
        train = (train - min_) / (max_ - min_) 
        
        bijection = []
        for i in range(len(self.topology)):
            if len(self.topology[i]) == 1:
                bijection.append(train[self.topology[i]].to_numpy())
                continue
            train_tmp = train[self.topology[i]].to_numpy()
            bijection_tmp = []
            for x, y in train_tmp:
                bijection_tmp.append(interleave_float(x, y))
            bijection.append(np.array([bijection_tmp]).T)
        bijection = np.concatenate(bijection, axis=1)
        self.label = bijection
            
        # from sklearn.decomposition import PCA
        # pca1 = PCA(n_components=1).fit_transform(df[self.topology[0]])
        # pca2 = PCA(n_components=1).fit_transform(df[self.topology[1]])
        # pca3 = PCA(n_components=1).fit_transform(df[self.topology[2]])
        # label = np.concatenate([pca1, pca2, pca3], axis=1)
        
        # """bounded label: normalize to (0, 1)"""
        # if config["label_normalization"]: 
        #     # label = (label - label.min(axis=0)) / (label.max(axis=0) - label.min(axis=0)) # global statistic
        #     label = (label - label.mean(axis=0)) / label.std(axis=0)
        # self.label = label[:4000, :]
        
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
        self.topology = [['Mortgage', 'Income'], ['Experience', 'Age'], ['CCAvg']]
        self.flatten_topology =  [self.continuous.index(item) for sublist in self.topology for item in sublist]
        
        df = df[self.continuous]
        
        min_ = df.min(axis=0)
        max_ = df.max(axis=0)
        test = df.iloc[4000:]
        test = (test - min_) / (max_ - min_) # local statistic
        
        bijection = []
        for i in range(len(self.topology)):
            if len(self.topology[i]) == 1:
                bijection.append(test[self.topology[i]].to_numpy())
                continue
            test_tmp = test[self.topology[i]].to_numpy()
            bijection_tmp = []
            for x, y in test_tmp:
                bijection_tmp.append(interleave_float(x, y))
            bijection.append(np.array([bijection_tmp]).T)
        bijection = np.concatenate(bijection, axis=1)
        self.label = bijection
            
        # from sklearn.decomposition import PCA
        # pca1 = PCA(n_components=1).fit_transform(df[self.topology[0]])
        # pca2 = PCA(n_components=1).fit_transform(df[self.topology[1]])
        # pca3 = PCA(n_components=1).fit_transform(df[self.topology[2]])
        # label = np.concatenate([pca1, pca2, pca3], axis=1)
        
        # """bounded label: normalize to (0, 1)"""
        # if config["label_normalization"]: 
        #     # label = (label - label.min(axis=0)) / (label.max(axis=0) - label.min(axis=0)) # global statistic
        #     label = (label - label.mean(axis=0)) / label.std(axis=0)
        # self.label = label[:4000, :]
        
        self.test = test
        self.x_data = test.to_numpy()
        
    def __len__(self): 
        return len(self.x_data)

    def __getitem__(self, idx): 
        x = torch.FloatTensor(self.x_data[idx])
        y = torch.FloatTensor(self.label[idx])
        return x, y
#%%