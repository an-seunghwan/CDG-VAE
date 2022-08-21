#%%
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
#%%
class VAE(nn.Module):
    def __init__(self, mask, config, device):
        super(VAE, self).__init__()
        
        self.config = config
        self.device = device
        
        """encoder"""
        self.encoder = nn.Sequential(
            nn.Linear(3*96*96, 300),
            nn.ELU(),
            nn.Linear(300, 300),
            nn.ELU(),
            nn.Linear(300, config["node"]),
            nn.BatchNorm1d(config["node"])
        ).to(device)
        
        self.mask = mask.to(device) 
        self.B = nn.Parameter(torch.randn(config["node"], config["node"]) * 0.1).to(device)
        
        """NPSEM"""
        self.npsem = [nn.Sequential(
            nn.Linear(config["node"] + 1, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
            nn.Tanh(),
            nn.Linear(2, 1),).to(device) 
            for _ in range(config["node"])]
        
        """decoder"""
        self.decoder = nn.Sequential(
            nn.Linear(config["node"], 300),
            nn.ELU(),
            nn.Linear(300, 300),
            nn.ELU(),
            nn.Linear(300, 3*96*96),
            nn.Tanh()
        ).to(device)
        
        # """Prior"""
        
    def forward(self, input):
        image, label = input
        logvar = self.encoder(nn.Flatten()(image)) # [batch, node * node_dim]
        
        """Latent Generating Process"""
        epsilon = torch.exp(logvar / 2) * torch.randn(image.size(0), self.config["node"]).to(self.device)
        B = self.mask * self.B
        align = torch.zeros((image.size(0), self.config["node"])).to(self.device)
        latent = torch.zeros((image.size(0), self.config["node"])).to(self.device)
        for j in range(self.config["node"]):
            align[:, [j]] = self.npsem[j](torch.cat((B[:, j] * latent, epsilon[:, [j]]), dim=1))
            latent[:, [j]] = torch.tanh(align[:, [j]])

        xhat = self.decoder(latent)
        xhat = xhat.view(-1, 96, 96, 3)

        """prior"""
        prior_logvar = label
        
        return logvar, prior_logvar, latent, align, xhat
#%%
def main():
    config = {
        "n": 10,
        "node": 4,
    }
    
    mask = torch.zeros(config["node"], config["node"])
    mask[:2, 2:] = 1
    mask[3, 2] = 1
    mask[2, 3] = 1
    model = VAE(mask, config, 'cpu')
    for x in model.parameters():
        print(x.shape)
        
    batch = torch.randn(config["n"], 96, 96, 3)
    u = torch.randn(config["n"], config["node"])
    logvar, prior_logvar, latent, align, xhat = model([batch, u])
    
    assert logvar.shape == (config["n"], config["node"])
    assert prior_logvar.shape == (config["n"], config["node"])
    assert latent.shape == (config["n"], config["node"])
    assert align.shape == (config["n"], config["node"])
    assert xhat.shape == (config["n"], 96, 96, 3)
    
    print("Model test pass!")
#%%
if __name__ == '__main__':
    main()
#%%
# config = {
#     "n": 10,
#     "node": 4,
# }

# x = torch.randn(config["n"], 96, 96, 3)
# u = torch.randn(config["n"], config["node"])

# config = config
# device = 'cpu'

# mask = torch.zeros(config["node"], config["node"])
# mask[:2, 2:] = 1
# mask[3, 2] = 1
# mask[2, 3] = 1

# B = nn.Parameter(torch.zeros(config["node"], config["node"]) * 0.1).to(device)
# B = B.to(device) # weighted adjacency matrix

# """encoder"""
# encoder = nn.Sequential(
#     nn.Linear(3*96*96, 300),
#     nn.ELU(),
#     nn.Linear(300, 300),
#     nn.ELU(),
#     nn.Linear(300, config["node"]),
# ).to(device)

# # batchnorm = nn.BatchNorm1d(config["node"] * config["embedding_dim"]).to(device)

# """NPSEM"""
# npsem = [nn.Sequential(
#     nn.Linear(config["node"] + 1, 4),
#     nn.Tanh(),
#     nn.Linear(4, 2),
#     nn.Tanh(),
#     nn.Linear(2, 1),).to(device) 
#     for _ in range(config["node"])]

# """decoder"""
# decoder = nn.Sequential(
#     nn.Linear(config["node"], 300),
#     nn.ELU(),
#     nn.Linear(300, 300),
#     nn.ELU(),
#     nn.Linear(300, 3*96*96),
#     nn.Tanh()
# ).to(device)
# #%%
# logvar = encoder(nn.Flatten()(x)) # [batch, node*embedding_dim]

# """Latent Generating Process"""
# epsilon = torch.exp(logvar / 2) * torch.randn(x.size(0), config["node"])

# align = torch.zeros((x.size(0), config["node"]))
# latent = torch.zeros((x.size(0), config["node"]))
# for j in range(config["node"]):
#     align[:, [j]] = npsem[j](torch.cat((B[:, j] * latent, epsilon[:, [j]]), dim=1))
#     latent[:, [j]] = torch.tanh(align[:, [j]])

# xhat = decoder(latent)
# xhat = xhat.view(-1, 96, 96, 3)

# """prior"""
# prior_logvar = u
# #%%