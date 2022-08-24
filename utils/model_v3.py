#%%
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
#%%
class VAE(nn.Module):
    def __init__(self, B, config, device):
        super(VAE, self).__init__()
        
        self.config = config
        self.device = device
        
        """encoder"""
        self.encoder = nn.Sequential(
            nn.Linear(3*96*96, 300),
            nn.ELU(),
            nn.Linear(300, 300),
            nn.ELU(),
            nn.Linear(300, config["node"] * config["node_dim"] * 2),
            nn.BatchNorm1d(config["node"] * config["node_dim"] * 2)
        ).to(device)
        
        self.B = B.to(device) # binary adjacency matrix
        
        """Addictive Noise Model"""
        self.lgp_nets = [nn.Sequential(
            nn.Linear(config["node"] * config["node_dim"], config["node"]),
            nn.ELU(),
            nn.Linear(config["node"], 2),
            nn.ELU(),
            nn.Linear(2, config["node_dim"]),
            nn.BatchNorm1d(config["node_dim"])
            ).to(device) for _ in range(config["node"])]
        
        """decoder"""
        self.decoder = nn.Sequential(
            nn.Linear(config["node"] * config["node_dim"], 300),
            nn.ELU(),
            nn.Linear(300, 300),
            nn.ELU(),
            nn.Linear(300, 3*96*96),
            nn.Tanh()
        ).to(device)
        
        """Alignment"""
        self.align_nets = [nn.Linear(config["node_dim"], 1).to(device) 
                           for _ in range(config["node"])]
        
        """Prior"""
        self.prior = [nn.Sequential(
            nn.Linear(1, 3),
            nn.ELU(),
            nn.Linear(3, config["node_dim"]),
            ).to(device) for _ in range(config["node"])]
        
    def forward(self, input):
        image, label = input
        h = self.encoder(nn.Flatten()(image)) # [batch, node * node_dim * 2]
        mean, logvar = torch.split(h, self.config["node"] * self.config["node_dim"], dim=1)
        
        """Latent Generating Process"""
        noise = torch.randn(image.size(0), self.config["node"] * self.config["node_dim"]).to(self.device) 
        epsilon = mean + torch.exp(logvar / 2) * noise
        # epsilon = epsilon.view(-1, self.config["node_dim"], self.config["node"]).contiguous()
        
        latent1 = torch.zeros(image.size(0), self.config["node"] * self.config["node_dim"]) # g(z)
        latent2 = torch.zeros(image.size(0), self.config["node"] * self.config["node_dim"]) # g(z) + e
        align_latent = []
        for j in range(self.config["node"]):
            child = [j + i * self.config["node"] for i in range(self.config["node_dim"])]
            h = self.lgp_nets[j](self.B[:, j].repeat(self.config["node_dim"]) * latent2)
            latent1[:, child] = h
            latent2[:, child] = h + epsilon[:, child]
            align_latent.append(self.align_nets[j](h + epsilon[:, child]))

        # intervention range
        latent3 = torch.tanh(latent2)

        xhat = self.decoder(latent3)
        xhat = xhat.view(-1, 96, 96, 3)

        """prior"""
        prior_logvar = list(map(lambda x, layer: layer(x), torch.split(label, 1, dim=1), self.prior))
        prior_logvar = torch.cat(prior_logvar, dim=1)
        
        return mean, logvar, prior_logvar, latent1, latent2, latent3, torch.cat(align_latent, dim=1), xhat
#%%
def main():
    config = {
        "n": 10,
        "node": 4,
        "node_dim": 2, 
    }
    
    B = torch.zeros(config["node"], config["node"])
    B[:2, 2:] = 1
    model = VAE(B, config, 'cpu')
    for x in model.parameters():
        print(x.shape)
        
    batch = torch.rand(config["n"], 96, 96, 3)
    u = torch.rand(config["n"], config["node"])
    mean, logvar, prior_logvar, latent1, latent2, latent3, align_latent, xhat = model([batch, u])
    
    assert mean.shape == (config["n"], config["node"] * config["node_dim"])
    assert logvar.shape == (config["n"], config["node"] * config["node_dim"])
    assert prior_logvar.shape == (config["n"], config["node"] * config["node_dim"])
    assert latent1.shape == (config["n"], config["node_dim"] * config["node"])
    assert latent2.shape == (config["n"], config["node_dim"] * config["node"])
    assert latent3.shape == (config["n"], config["node_dim"] * config["node"])
    assert align_latent.shape == (config["n"], config["node"])
    assert xhat.shape == (config["n"], 96, 96, 3)
    
    print("Model test pass!")
#%%
if __name__ == '__main__':
    main()
#%%
# config = {
#     "n": 10,
#     "node": 4,
#     "node_dim": 2,
# }

# B = torch.zeros(config["node"], config["node"])
# B[:2, 2:] = 1

# x = torch.randn(config["n"], 96, 96, 3)
# u = torch.randn(config["n"], config["node"])

# config = config
# device = 'cpu'

# """encoder"""
# encoder = nn.Sequential(
#     nn.Linear(3*96*96, 300),
#     nn.ELU(),
#     nn.Linear(300, 300),
#     nn.ELU(),
#     nn.Linear(300, config["node"] * config["node_dim"] * 2),
# ).to(device)

# B = B.to(device) # weighted adjacency matrix
# # batchnorm = nn.BatchNorm1d(config["node"] * config["embedding_dim"]).to(device)

# lgp_nets = [nn.Sequential(
#     nn.Linear(config["node"] * config["node_dim"], config["node"]),
#     nn.ELU(),
#     nn.Linear(config["node"], 2),
#     nn.ELU(),
#     nn.Linear(2, config["node_dim"]),
#     ).to(device) for _ in range(config["node"])]

# """Alignment"""
# align_nets = [nn.Linear(config["node_dim"], 1).to(device) for _ in range(config["node"])]

# """decoder"""
# decoder = nn.Sequential(
#     nn.Linear(config["node"] * config["node_dim"], 300),
#     nn.ELU(),
#     nn.Linear(300, 300),
#     nn.ELU(),
#     nn.Linear(300, 3*96*96),
#     nn.Tanh()
# ).to(device)

# I = torch.eye(config["node"]).to(device)

# """Prior"""
# prior = [nn.Sequential(
#     nn.Linear(1, 3),
#     nn.ELU(),
#     nn.Linear(3, config["node_dim"]),
#     ).to(device) for _ in range(config["node"])]
# #%%
# h = encoder(nn.Flatten()(x)) # [batch, node * node_dim * 2]
# mean, logvar = torch.split(h, config["node"] * config["node_dim"], dim=1)

# """Latent Generating Process"""
# noise = torch.randn(x.size(0), config["node"] * config["node_dim"]).to(device) 
# epsilon = mean + torch.exp(logvar / 2) * noise
# # epsilon = epsilon.view(-1, config["node"], config["node_dim"]).contiguous()

# latent1 = torch.zeros(x.size(0), config["node"] * config["node_dim"]) # g(z)
# latent2 = torch.zeros(x.size(0), config["node"] * config["node_dim"]) # g(z) + e
# latent3 = torch.zeros(x.size(0), config["node"] * config["node_dim"]) # tanh(g(z) + e)
# align_latent = []
# for j in range(config["node"]):
#     child = [j + i * config["node"] for i in range(config["node_dim"])]
#     h = lgp_nets[j](B[:, j].repeat(config["node_dim"]) * latent2)
#     latent1[:, child] = h
#     latent2[:, child] = h + epsilon[:, child]
#     latent3[:, child] = torch.tanh(h + epsilon[:, child])
#     align_latent.append(align_nets[j](h + epsilon[:, child]))

# xhat = decoder(latent3)
# xhat = xhat.view(-1, 96, 96, 3)

# """prior"""
# prior_logvar = list(map(lambda x, layer: layer(x), torch.split(u, 1, dim=1), prior))
# prior_logvar = torch.cat(prior_logvar, dim=1)
#%%