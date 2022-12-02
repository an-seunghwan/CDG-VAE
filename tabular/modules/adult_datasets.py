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

from modules.data_transformer import DataTransformer
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
        if config["dataset"] == 'loan':
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
            
            df_ = (df - df.mean(axis=0)) / df.std(axis=0)
            train = df_.iloc[:4000]
            
            min_ = df_.min(axis=0)
            max_ = df_.max(axis=0)
            df = (df_ - min_) / (max_ - min_) 
            
            bijection = []
            for i in range(len(self.topology)):
                if len(self.topology[i]) == 1:
                    bijection.append(df[self.topology[i]].to_numpy())
                    continue
                df_tmp = df[self.topology[i]].to_numpy()
                bijection_tmp = []
                for x, y in df_tmp:
                    bijection_tmp.append(interleave_float(x, y))
                bijection.append(np.array([bijection_tmp]).T)
            bijection = np.concatenate(bijection, axis=1)
            self.label = bijection[:4000, :]
                
            self.train = train
            self.x_data = train.to_numpy()
            
        elif config["dataset"] == 'adult':
            """
            load dataset: Adult
            Reference: https://archive.ics.uci.edu/ml/datasets/Adult
            """
            df = pd.read_csv('./data/adult.csv')
            df = df.sample(frac=1, random_state=1).reset_index(drop=True)
            df = df[(df == '?').sum(axis=1) == 0]
            df['income'] = df['income'].map({'<=50K': 0, '>50K': 1, '<=50K.': 0, '>50K.': 1})
            
            self.continuous = ['income', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week']
            self.topology = [['capital-gain'], ['capital-loss'], ['income', 'educational-num', 'hours-per-week']]
            self.flatten_topology =  [self.continuous.index(item) for sublist in self.topology for item in sublist]
            df = df[self.continuous]
            
            scaling = [x for x in self.continuous if x != 'income']
            df_ = df.copy()
            df_[scaling] = (df[scaling] - df[scaling].mean(axis=0)) / df[scaling].std(axis=0)
            # df_ = (df - df.mean(axis=0)) / df.std(axis=0)
            train = df_.iloc[:40000, ]
            
            min_ = df_.min(axis=0)
            max_ = df_.max(axis=0)
            df = (df_ - min_) / (max_ - min_) 
            
            bijection = []
            for i in range(len(self.topology)):
                if len(self.topology[i]) == 1:
                    bijection.append(df[self.topology[i]].to_numpy())
                    continue
                if len(self.topology[i]) == 3:
                    df_tmp = df[self.topology[i]].to_numpy()
                    bijection_tmp = []
                    for x, y in df_tmp[:, :2]:
                        bijection_tmp.append(interleave_float(x, y))
                    tmp = np.concatenate([np.array([bijection_tmp]).T, df_tmp[:, [2]]], axis=1)
                    bijection_tmp = []
                    for x, y in tmp:
                        bijection_tmp.append(interleave_float(x, y))
                    bijection.append(np.array([bijection_tmp]).T)
            bijection = np.concatenate(bijection, axis=1)
            self.label = bijection[:40000, ]
            
            self.train = train
            self.x_data = train.to_numpy()
        
        elif config["dataset"] == 'credit':
            base = pd.read_csv('./data/home-credit-default-risk/application_train.csv')
            base = base.sample(frac=1, random_state=1).reset_index(drop=True)
            
            self.continuous = ['TARGET', 'AMT_ANNUITY', 'AMT_CREDIT', 'DAYS_EMPLOYED', 'DAYS_BIRTH']
            self.topology = [['TARGET'], ['AMT_ANNUITY'], ['AMT_CREDIT', 'DAYS_EMPLOYED', 'DAYS_BIRTH']]
            df = base[self.continuous]
            df = df.dropna(axis=0)
            # imbalanced class
            np.random.seed(1)
            idx = np.random.choice(
                range((df['TARGET'] == 0).sum()), 
                (df['TARGET'] == 1).sum() * 1, 
                replace=False)
            df = pd.concat([df.iloc[idx], df[df['TARGET'] == 1]], axis=0).reset_index().drop(columns=['index'])
            df = df.sample(frac=1, random_state=1).reset_index(drop=True)
            
            """remove outlier"""
            for col in self.continuous:
                if col == 'TARGET': continue
                
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1

                condition = df[col] > q3 + 1.5 * iqr
                idx = df[condition].index
                df.drop(idx, inplace=True)
                condition = df[col] < q1 - 1.5 * iqr
                idx = df[condition].index
                df.drop(idx, inplace=True)
            
            scaling = [x for x in self.continuous if x != 'TARGET']
            df_ = df.copy()
            df_[scaling] = (df[scaling] - df[scaling].mean(axis=0)) / df[scaling].std(axis=0)
            train = df_.iloc[:30000, ]
            
            min_ = df_.min(axis=0)
            max_ = df_.max(axis=0)
            df = (df_ - min_) / (max_ - min_) 
            
            bijection = []
            for i in range(len(self.topology)):
                if len(self.topology[i]) == 1:
                    bijection.append(df[self.topology[i]].to_numpy())
                    continue
                if len(self.topology[i]) == 2:
                    df_tmp = df[self.topology[i]].to_numpy()
                    bijection_tmp = []
                    for x, y in df_tmp:
                        bijection_tmp.append(interleave_float(x, y))
                    bijection.append(np.array([bijection_tmp]).T)
                if len(self.topology[i]) == 3:
                    df_tmp = df[self.topology[i]].to_numpy()
                    bijection_tmp = []
                    for x, y in df_tmp[:, :2]:
                        bijection_tmp.append(interleave_float(x, y))
                    tmp = np.concatenate([np.array([bijection_tmp]).T, df_tmp[:, [2]]], axis=1)
                    bijection_tmp = []
                    for x, y in tmp:
                        bijection_tmp.append(interleave_float(x, y))
                    bijection.append(np.array([bijection_tmp]).T)
            bijection = np.concatenate(bijection, axis=1)
            self.label = bijection[:30000, ]
            
            self.train = train
            self.x_data = train.to_numpy()
            
        else:
            raise ValueError('Not supported dataset!')
        
    def __len__(self): 
        return len(self.x_data)

    def __getitem__(self, idx): 
        x = torch.FloatTensor(self.x_data[idx])
        y = torch.FloatTensor(self.label[idx])
        return x, y
#%%
class TestTabularDataset(Dataset): 
    def __init__(self, config):
        if config["dataset"] == 'loan':
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
            
            df_ = (df - df.mean(axis=0)) / df.std(axis=0)
            test = df_.iloc[4000:]
            
            min_ = df_.min(axis=0)
            max_ = df_.max(axis=0)
            df = (df_ - min_) / (max_ - min_) 
            
            bijection = []
            for i in range(len(self.topology)):
                if len(self.topology[i]) == 1:
                    bijection.append(df[self.topology[i]].to_numpy())
                    continue
                df_tmp = df[self.topology[i]].to_numpy()
                bijection_tmp = []
                for x, y in df_tmp:
                    bijection_tmp.append(interleave_float(x, y))
                bijection.append(np.array([bijection_tmp]).T)
            bijection = np.concatenate(bijection, axis=1)
            self.label = bijection[4000:, :]
                
            self.test = test
            self.x_data = test.to_numpy()
        
        elif config["dataset"] == 'adult':
            """
            load dataset: Adult
            Reference: https://archive.ics.uci.edu/ml/datasets/Adult
            """
            df = pd.read_csv('./data/adult.csv')
            df = df.sample(frac=1, random_state=1).reset_index(drop=True)
            df = df[(df == '?').sum(axis=1) == 0]
            df['income'] = df['income'].map({'<=50K': 0, '>50K': 1, '<=50K.': 0, '>50K.': 1})
            
            self.continuous = ['income', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week']
            self.topology = [['capital-gain'], ['capital-loss'], ['income', 'educational-num', 'hours-per-week']]
            self.flatten_topology =  [self.continuous.index(item) for sublist in self.topology for item in sublist]
            df = df[self.continuous]
            
            scaling = [x for x in self.continuous if x != 'income']
            df_ = df.copy()
            df_[scaling] = (df[scaling] - df[scaling].mean(axis=0)) / df[scaling].std(axis=0)
            # df_ = (df - df.mean(axis=0)) / df.std(axis=0)
            test = df_.iloc[40000:, ]
            
            min_ = df_.min(axis=0)
            max_ = df_.max(axis=0)
            df = (df_ - min_) / (max_ - min_) 
            
            bijection = []
            for i in range(len(self.topology)):
                if len(self.topology[i]) == 1:
                    bijection.append(df[self.topology[i]].to_numpy())
                    continue
                if len(self.topology[i]) == 3:
                    df_tmp = df[self.topology[i]].to_numpy()
                    bijection_tmp = []
                    for x, y in df_tmp[:, :2]:
                        bijection_tmp.append(interleave_float(x, y))
                    tmp = np.concatenate([np.array([bijection_tmp]).T, df_tmp[:, [2]]], axis=1)
                    bijection_tmp = []
                    for x, y in tmp:
                        bijection_tmp.append(interleave_float(x, y))
                    bijection.append(np.array([bijection_tmp]).T)
            bijection = np.concatenate(bijection, axis=1)
            self.label = bijection[40000:, ]
            
            self.test = test
            self.x_data = test.to_numpy()
        
        elif config["dataset"] == 'credit':
            base = pd.read_csv('./data/home-credit-default-risk/application_train.csv')
            base = base.sample(frac=1, random_state=1).reset_index(drop=True)
            
            self.continuous = ['TARGET', 'AMT_ANNUITY', 'AMT_CREDIT', 'DAYS_EMPLOYED', 'DAYS_BIRTH']
            self.topology = [['TARGET'], ['AMT_ANNUITY'], ['AMT_CREDIT', 'DAYS_EMPLOYED', 'DAYS_BIRTH']]
            df = base[self.continuous]
            df = df.dropna(axis=0)
            # imbalanced class
            np.random.seed(1)
            idx = np.random.choice(
                range((df['TARGET'] == 0).sum()), 
                (df['TARGET'] == 1).sum() * 1, 
                replace=False)
            df = pd.concat([df.iloc[idx], df[df['TARGET'] == 1]], axis=0).reset_index().drop(columns=['index'])
            df = df.sample(frac=1, random_state=1).reset_index(drop=True)
            
            """remove outlier"""
            for col in self.continuous:
                if col == 'TARGET': continue
                
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1

                condition = df[col] > q3 + 1.5 * iqr
                idx = df[condition].index
                df.drop(idx, inplace=True)
                condition = df[col] < q1 - 1.5 * iqr
                idx = df[condition].index
                df.drop(idx, inplace=True)
            
            scaling = [x for x in self.continuous if x != 'TARGET']
            df_ = df.copy()
            df_[scaling] = (df[scaling] - df[scaling].mean(axis=0)) / df[scaling].std(axis=0)
            test = df_.iloc[30000:, ]
            
            min_ = df_.min(axis=0)
            max_ = df_.max(axis=0)
            df = (df_ - min_) / (max_ - min_) 
            
            bijection = []
            for i in range(len(self.topology)):
                if len(self.topology[i]) == 1:
                    bijection.append(df[self.topology[i]].to_numpy())
                    continue
                if len(self.topology[i]) == 2:
                    df_tmp = df[self.topology[i]].to_numpy()
                    bijection_tmp = []
                    for x, y in df_tmp:
                        bijection_tmp.append(interleave_float(x, y))
                    bijection.append(np.array([bijection_tmp]).T)
                if len(self.topology[i]) == 3:
                    df_tmp = df[self.topology[i]].to_numpy()
                    bijection_tmp = []
                    for x, y in df_tmp[:, :2]:
                        bijection_tmp.append(interleave_float(x, y))
                    tmp = np.concatenate([np.array([bijection_tmp]).T, df_tmp[:, [2]]], axis=1)
                    bijection_tmp = []
                    for x, y in tmp:
                        bijection_tmp.append(interleave_float(x, y))
                    bijection.append(np.array([bijection_tmp]).T)
            bijection = np.concatenate(bijection, axis=1)
            self.label = bijection[30000:, ]
            
            self.test = test
            self.x_data = test.to_numpy()
            
        else:
            raise ValueError('Not supported dataset!')
        
    def __len__(self): 
        return len(self.x_data)

    def __getitem__(self, idx): 
        x = torch.FloatTensor(self.x_data[idx])
        y = torch.FloatTensor(self.label[idx])
        return x, y
#%%
class TabularDataset2(Dataset): 
    def __init__(self, config):
        if config["dataset"] == 'loan':
            """
            load dataset: Personal Loan
            Reference: https://www.kaggle.com/datasets/teertha/personal-loan-modeling
            """
            df = pd.read_csv('./data/Bank_Personal_Loan_Modelling.csv')
            df = df.sample(frac=1, random_state=1).reset_index(drop=True)
            df = df.drop(columns=['ID'])
            self.continuous = ['Mortgage', 'Income', 'Experience', 'Age', 'CCAvg']
            df = df[self.continuous]
            self.topology = [['Mortgage', 'Income'], ['Experience', 'Age'], ['CCAvg']]
            
            min_ = df.min(axis=0)
            max_ = df.max(axis=0)
            df_normalized = (df - min_) / (max_ - min_) 
            
            bijection = []
            for i in range(len(self.topology)):
                if len(self.topology[i]) == 1:
                    bijection.append(df_normalized[self.topology[i]].to_numpy())
                    continue
                df_tmp = df_normalized[self.topology[i]].to_numpy()
                bijection_tmp = []
                for x, y in df_tmp:
                    bijection_tmp.append(interleave_float(x, y))
                bijection.append(np.array([bijection_tmp]).T)
            bijection = np.concatenate(bijection, axis=1)
            self.label = bijection[:4000, :]
            
            df = df[self.continuous].iloc[:4000]
            
            transformer = DataTransformer()
            transformer.fit(df)
            train_data = transformer.transform(df)
            self.transformer = transformer
                
            self.train = train_data
            self.x_data = torch.from_numpy(train_data.astype('float32'))
        
        elif config["dataset"] == 'adult':
            """
            load dataset: Adult
            Reference: https://archive.ics.uci.edu/ml/datasets/Adult
            """
            df = pd.read_csv('./data/adult.csv')
            df = df.sample(frac=1, random_state=1).reset_index(drop=True)
            df = df[(df == '?').sum(axis=1) == 0]
            df['income'] = df['income'].map({'<=50K': 0, '>50K': 1, '<=50K.': 0, '>50K.': 1})
            
            self.continuous = ['income', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week']
            self.topology = [['capital-gain'], ['capital-loss'], ['income', 'educational-num', 'hours-per-week']]
            self.flatten_topology =  [self.continuous.index(item) for sublist in self.topology for item in sublist]
            df = df[self.continuous]
            
            min_ = df.min(axis=0)
            max_ = df.max(axis=0)
            df_normalized = (df - min_) / (max_ - min_) 
            
            bijection = []
            for i in range(len(self.topology)):
                if len(self.topology[i]) == 1:
                    bijection.append(df_normalized[self.topology[i]].to_numpy())
                    continue
                if len(self.topology[i]) == 3:
                    df_tmp = df_normalized[self.topology[i]].to_numpy()
                    bijection_tmp = []
                    for x, y in df_tmp[:, :2]:
                        bijection_tmp.append(interleave_float(x, y))
                    tmp = np.concatenate([np.array([bijection_tmp]).T, df_tmp[:, [2]]], axis=1)
                    bijection_tmp = []
                    for x, y in tmp:
                        bijection_tmp.append(interleave_float(x, y))
                    bijection.append(np.array([bijection_tmp]).T)
            bijection = np.concatenate(bijection, axis=1)
            self.label = bijection[:40000, ]
            
            df = df[self.continuous].iloc[:4000]
            
            transformer = DataTransformer()
            transformer.fit(df)
            train_data = transformer.transform(df)
            self.transformer = transformer
                
            self.train = train_data
            self.x_data = torch.from_numpy(train_data.astype('float32'))
        
        elif config["dataset"] == 'credit':
            base = pd.read_csv('./data/home-credit-default-risk/application_train.csv')
            base = base.sample(frac=1, random_state=1).reset_index(drop=True)
            
            self.continuous = ['TARGET', 'AMT_ANNUITY', 'AMT_CREDIT', 'DAYS_EMPLOYED', 'DAYS_BIRTH']
            self.topology = [['TARGET'], ['AMT_ANNUITY'], ['AMT_CREDIT', 'DAYS_EMPLOYED', 'DAYS_BIRTH']]
            df = base[self.continuous]
            df = df.dropna(axis=0)
            # imbalanced class
            np.random.seed(1)
            idx = np.random.choice(
                range((df['TARGET'] == 0).sum()), 
                (df['TARGET'] == 1).sum() * 1, 
                replace=False)
            df = pd.concat([df.iloc[idx], df[df['TARGET'] == 1]], axis=0).reset_index().drop(columns=['index'])
            df = df.sample(frac=1, random_state=1).reset_index(drop=True)
            
            """remove outlier"""
            for col in self.continuous:
                if col == 'TARGET': continue
                
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1

                condition = df[col] > q3 + 1.5 * iqr
                idx = df[condition].index
                df.drop(idx, inplace=True)
                condition = df[col] < q1 - 1.5 * iqr
                idx = df[condition].index
                df.drop(idx, inplace=True)
            
            min_ = df.min(axis=0)
            max_ = df.max(axis=0)
            df_normalized = (df - min_) / (max_ - min_) 
            
            bijection = []
            for i in range(len(self.topology)):
                if len(self.topology[i]) == 1:
                    bijection.append(df_normalized[self.topology[i]].to_numpy())
                    continue
                if len(self.topology[i]) == 2:
                    df_tmp = df_normalized[self.topology[i]].to_numpy()
                    bijection_tmp = []
                    for x, y in df_tmp:
                        bijection_tmp.append(interleave_float(x, y))
                    bijection.append(np.array([bijection_tmp]).T)
                if len(self.topology[i]) == 3:
                    df_tmp = df_normalized[self.topology[i]].to_numpy()
                    bijection_tmp = []
                    for x, y in df_tmp[:, :2]:
                        bijection_tmp.append(interleave_float(x, y))
                    tmp = np.concatenate([np.array([bijection_tmp]).T, df_tmp[:, [2]]], axis=1)
                    bijection_tmp = []
                    for x, y in tmp:
                        bijection_tmp.append(interleave_float(x, y))
                    bijection.append(np.array([bijection_tmp]).T)
            bijection = np.concatenate(bijection, axis=1)
            self.label = bijection[:30000, ]
            
            df = df.iloc[:30000]
            
            transformer = DataTransformer()
            transformer.fit(df)
            train_data = transformer.transform(df)
            self.transformer = transformer
                
            self.train = train_data
            self.x_data = torch.from_numpy(train_data.astype('float32'))
            
        else:
            raise ValueError('Not supported dataset!')
        
    def __len__(self): 
        return len(self.x_data)

    def __getitem__(self, idx): 
        x = torch.FloatTensor(self.x_data[idx])
        y = torch.FloatTensor(self.label[idx])
        return x, y
#%%