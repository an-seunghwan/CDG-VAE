#%%
import numpy as np
import torch
import matplotlib.pyplot as plt
from model_0 import PlanarFlows
#%%
dim = 1
flow_num = 10
inverse_loop = 100
device = 'cpu'
n = 1000
#%%
torch.manual_seed(1)
base = torch.randn(n, 1)
flow = PlanarFlows(1, flow_num, inverse_loop, device)
z = flow(base)
#%%
plt.hist(base.numpy(), bins=int(np.sqrt(n)))
plt.hist(z.detach().numpy(), bins=int(np.sqrt(n)))
#%%