#%%
#Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
#This program is free software; 
#you can redistribute it and/or modify
#it under the terms of the MIT License.
#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the MIT License for more details.
#%%
import torch
from torch import nn
#%% 
class MaskLayer(nn.Module):
    def __init__(self, concept=4):
        super().__init__()
        self.concept = concept

        self.elu = nn.ELU()
        self.net1 = nn.Sequential(
        nn.Linear(1 , 32),
        nn.ELU(),
        nn.Linear(32, 1),
        )
        self.net2 = nn.Sequential(
        nn.Linear(1 , 32),
        nn.ELU(),
        nn.Linear(32, 1),
        )
        self.net3 = nn.Sequential(
        nn.Linear(1 , 32),
        nn.ELU(),
        nn.Linear(32, 1),
        )
        self.net4 = nn.Sequential(
        nn.Linear(1 , 32),
        nn.ELU(),
        nn.Linear(32, 1)
        )

    def forward(self, z):
        z = z.view(-1, self.concept, 1)
        z1, z2, z3, z4= z[:, 0], z[:, 1], z[:, 2], z[:, 3]

        rx1 = self.net1(z1)
        rx2 = self.net2(z2)
        rx3 = self.net3(z3)
        rx4 = self.net4(z4)

        h = torch.cat((rx1,rx2,rx3,rx4), dim=1)

        return h
#%%
# model = MaskLayer()
# x = torch.randn(10, 4)
# model(x)
#%%