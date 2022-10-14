#%%
import numpy as np
import pandas as pd
import statsmodels.api as sm
import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import Dataset
#%%
np.random.seed(1)
n = 10000

"""ground-truth causal graph"""
gamma = 1
beta = 2

x1 = np.random.normal(size=(n, 1))
z1 = gamma * x1
y = np.random.binomial(n=1, p=1 / (1 + np.exp(-beta * z1)))

x2 = np.zeros((n, 1))
ratio = 0.9
x2[np.where(y==1)[0][:int(ratio*len(np.where(y==1)[0]))]] = 2
x2[np.where(y==1)[0][int(ratio*len(np.where(y==1)[0])):]] = -2
x2[np.where(y==0)[0][:int(ratio*len(np.where(y==0)[0]))]] = -2
x2[np.where(y==0)[0][int(ratio*len(np.where(y==0)[0])):]] = 2
x2 = np.random.normal(loc=x2)

alpha = 1
z2 = alpha * (x2 > 0) + x2

print('causal correlation: {:.3f}'.format(np.corrcoef(z1[:, 0], y[:, 0])[0, 1]))
print('spurious correlation: {:.3f}'.format(np.corrcoef(x2[:, 0], y[:, 0])[0, 1]))
#%%
# test dataset
test_x1 = np.random.normal(size=(n, 1))
test_z1 = gamma * test_x1 
test_y = np.random.binomial(n=1, p=1 / (1 + np.exp(-beta * test_z1)))

test_x2 = np.zeros((n, 1))
ratio = 0.5
test_x2[np.where(test_y==1)[0][:int(ratio*len(np.where(test_y==1)[0]))]] = 2
test_x2[np.where(test_y==1)[0][int(ratio*len(np.where(test_y==1)[0])):]] = -2
test_x2[np.where(test_y==0)[0][:int(ratio*len(np.where(test_y==0)[0]))]] = -2
test_x2[np.where(test_y==0)[0][int(ratio*len(np.where(test_y==0)[0])):]] = 2
test_x2 = np.random.normal(loc=test_x2)

test_z2 = alpha * (test_x2 > 0) + test_x2

print('test causal correlation: {:.3f}'.format(np.corrcoef(test_z1[:, 0], test_y[:, 0])[0, 1]))
print('test spurious correlation: {:.3f}'.format(np.corrcoef(test_x2[:, 0], test_y[:, 0])[0, 1]))
#%%
x = np.concatenate([x1, x2], axis=1)
z = np.concatenate([z1, z2], axis=1)
test_x = np.concatenate([test_x1, test_x2], axis=1)
test_z = np.concatenate([test_z1, test_z2], axis=1)
#%%
"""Disentangled representation model"""
gtmodel = sm.Logit(y, z1).fit()
print(gtmodel.summary())

gtpred = gtmodel.predict(test_z1)
gtacc = ((gtpred > 0.5).astype(float) == test_y.squeeze()).mean()
#%%
"""ERM"""
ERMmodel = sm.Logit(y, x).fit()
print(ERMmodel.summary())

ERMpred = ERMmodel.predict(test_x)
ERMacc = ((ERMpred > 0.5).astype(float) == test_y.squeeze()).mean()
#%%
x = torch.tensor(x, dtype=torch.float32)
z = torch.tensor(z, dtype=torch.float32)
# z = (z - z.min(axis=0).values) / (z.max(axis=0).values - z.min(axis=0).values)
y = torch.tensor(y)

test_x = torch.tensor(test_x, dtype=torch.float32)
test_z = torch.tensor(test_z, dtype=torch.float32)
# test_z = (test_z - test_z.min(axis=0).values) / (test_z.max(axis=0).values - test_z.min(axis=0).values)
test_y = torch.tensor(test_y)

dataset = TensorDataset(x, z, y)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
test_dataset = TensorDataset(test_x, test_z, test_y)
test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=True)
#%%
"""Entangled representation"""
class EntangledModel(nn.Module):
    def __init__(self):
        super(EntangledModel, self).__init__()
        self.fc1 = nn.Linear(2, 1, bias=False)
        self.fc2 = nn.Linear(1, 1, bias=False)
        
    def forward(self, input):
        z = self.fc1(input)
        out = torch.sigmoid(self.fc2(z))
        return z, out

torch.manual_seed(1)
model = EntangledModel()
optimizer = torch.optim.Adam(
    model.parameters(), 
    lr=0.005
)
model.train()
for epoch in range(20):
    for (x_batch, z_batch, y_batch) in iter(dataloader):
        optimizer.zero_grad()
        z, pred = model(x_batch)
        loss = -(y_batch * torch.log(pred) + (1 - y_batch) * torch.log(1 - pred)).mean()
        # loss += -(z_batch * torch.log(torch.sigmoid(z)) + (1 - z_batch) * torch.log(1 - torch.sigmoid(z))).mean()
        loss.backward()
        optimizer.step()
            
    # accuracy
    with torch.no_grad():
        """train accuracy"""
        train_correct = 0
        count = 0
        for (x_batch, z_batch, y_batch) in iter(dataloader):
            z, pred = model(x_batch)
            train_correct += ((pred > 0.5).float() == y_batch).float().sum().item()
            count += pred.size(0)
        train_correct /= count
        
        """test accuracy"""
        test_correct = 0
        count = 0
        for (x_batch, z_batch, y_batch) in iter(test_dataloader):
            z, pred = model(x_batch)
            test_correct += ((pred > 0.5).float() == y_batch).float().sum().item()
            count += pred.size(0)
        test_correct /= count

    print_input = "[EPOCH {:03d}]".format(epoch + 1)
    print_input += ' Loss: {:.4f}'.format(loss)
    print_input += ', TrainACC: {:.2f}%'.format(train_correct * 100)
    print_input += ', TestACC: {:.2f}%'.format(test_correct * 100)
    print(print_input)
#%%
print('Disentangled model test accuracy: {:.2f}%'.format(gtacc * 100))
print('ERM model test accuracy: {:.2f}%'.format(ERMacc * 100))
print('Entangled model test accuracy: {:.2f}%'.format(test_correct * 100))
#%%