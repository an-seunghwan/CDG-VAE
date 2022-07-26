#%%
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
#%%
class VAE(nn.Module):
    def __init__(self, config, device):
        super(VAE, self).__init__()
        
        self.config = config
        self.device = device
        
        """encoder"""
        self.encoder = nn.Sequential(
            nn.Linear(3*96*96, 300),
            nn.ELU(),
            nn.Linear(300, 300),
            nn.ELU(),
            nn.Linear(300, config["latent_dim"]),
        ).to(device)

        """logit of adjacency matrix"""
        p = {x:y for x,y in zip(range(config["latent_dim"]), range(config["latent_dim"]))}
        # build ReLU(Y)
        Y = torch.zeros((config["latent_dim"], config["latent_dim"]))
        for i in range(config["latent_dim"]):
            for j in range(config["latent_dim"]):
                Y[i, j] = (p[j] - p[i])
                # if i != j:
                #     Y[i, j] = (p[j] - p[i]) / np.abs(p[j] - p[i])
        self.ReLU_Y = torch.nn.ReLU()(Y).to(device)

        self.W = nn.Parameter(
                torch.zeros((config["latent_dim"], config["latent_dim"]), 
                            requires_grad=True).to(device)) # logit

        """decoder"""
        self.decoder = nn.Sequential(
            nn.Linear(config["latent_dim"], 300),
            nn.ELU(),
            nn.Linear(300, 300),
            nn.ELU(),
            nn.Linear(300, 3*96*96),
            nn.Tanh()
        ).to(device)

        # self.tau = config["temperature"]
        self.I = torch.eye(config["latent_dim"]).to(device)
    
    # def sample_gumbel(self, shape, eps=1e-20):
    #     U = torch.rand(shape).to(self.device)
    #     g1 = -torch.log(-torch.log(U + eps) + eps)
    #     U = torch.rand(shape).to(self.device)
    #     g2 = -torch.log(-torch.log(U + eps) + eps)
    #     return g1 - g2
    
    def forward(self, input):
        # h = self.encoder(nn.Flatten()(input))
        # exog_mean, exog_logvar = torch.split(h, self.config["latent_dim"], dim=1)
        exog = self.encoder(nn.Flatten()(input))
        # exog_logvar = torch.tanh(exog_logvar) * 0.5 # variance scaling (exp(-0.5) ~ exp(0.5))

        """Latent Generating Process"""
        # B = torch.sigmoid((self.W + self.sample_gumbel(self.W.shape)) / self.tau) * self.ReLU_Y # B \in {0, 1}
        # B = torch.sigmoid(self.W) * self.ReLU_Y # B \in [0, 1]
        B = self.W * self.ReLU_Y # B \in [-inf, inf]
        
        # fixed variance for 0.1 -> prevent epsilon dominating in LGP
        epsilon = exog + 0.3 * torch.randn(exog.shape).to(self.device) 
        # epsilon = exog_mean + torch.exp(exog_logvar / 2) * torch.randn(exog_mean.shape).to(self.device)
        latent = torch.tanh(torch.matmul(epsilon, torch.inverse(self.I - B)))

        # inverse_tanh_latent = 0.5 * torch.log((1. + latent) / (1. - latent) + 1e-8)

        # 1. with z
        xhat = self.decoder(latent)
        # 2. with Bz
        # xhat = self.decoder(torch.matmul(latent, B))
        # 3. with Bz + epsilon (SEM)
        # xhat = self.decoder(torch.matmul(latent, B) + epsilon)

        xhat = xhat.view(-1, 96, 96, 3)
        return exog, latent, B, xhat
#%%
def main():
    config = {
        "n": 100,
        "latent_dim": 5,
        "temperature": 0.2
    }
    
    model = VAE(config, 'cpu')
    for x in model.parameters():
        print(x.shape)
        
    batch = torch.rand(config["n"], 96, 96, 3)
    exog_mean, exog_logvar, inverse_sigmoid_latent, B, xhat = model(batch)
    
    assert exog_mean.shape == (config["n"], config["latent_dim"])
    assert exog_logvar.shape == (config["n"], config["latent_dim"])
    assert inverse_sigmoid_latent.shape == (config["n"], config["latent_dim"])
    assert B.shape == (config["latent_dim"], config["latent_dim"])
    assert xhat.shape == (config["n"], 96, 96, 3)
    
    print("Model test pass!")
#%%
if __name__ == '__main__':
    main()
#%%
# from PIL import Image
# img = Image.open("/Users/anseunghwan/Documents/GitHub/causal_vae/utils/causal_data/pendulum/train/a_-1_60_6_5.png")
# np.array(img)[:, :, :3]
#%%
# """encoder"""
# encoder = nn.Sequential(
#     nn.Linear(3*96*96, 300),
#     nn.ELU(),
#     nn.Linear(300, 300),
#     nn.ELU(),
#     nn.Linear(300, 2 * config["latent_dim"]),
# )

# """logit of adjacency matrix"""
# p = {x:y for x,y in zip(range(config["latent_dim"]), range(config["latent_dim"]))}
# # build ReLU(Y)
# Y = torch.zeros((config["latent_dim"], config["latent_dim"]))
# for i in range(config["latent_dim"]):
#     for j in range(config["latent_dim"]):
#         if i != j:
#             Y[i, j] = (p[j] - p[i]) / np.abs(p[j] - p[i])
# ReLU_Y = torch.nn.ReLU()(Y)

# torch.inverse(I - ReLU_Y * 0.01``)

# W = nn.Parameter(
#         torch.zeros((config["latent_dim"], config["latent_dim"]), 
#                     requires_grad=True)) # logit

# """decoder"""
# decoder = nn.Sequential(
#     nn.Linear(config["latent_dim"], 300),
#     nn.ELU(),
#     nn.Linear(300, 900),
#     nn.ELU(),
#     nn.Linear(900, 3*96*96),
#     nn.Tanh()
# )

# tau = config["temperature"]
# I = torch.eye(config["latent_dim"])

# input = torch.randn(10, 96, 96, 3)

# h = encoder(nn.Flatten()(input))
# exog_mean, exog_logvar = torch.split(h, config["latent_dim"], dim=1)
# # exog_logvar = torch.tanh(exog_logvar) * 0.5 # variance scaling (exp(-0.5) ~ exp(0.5))

# def sample_gumbel(shape, eps=1e-20):
#     U = torch.rand(shape)
#     g1 = -torch.log(-torch.log(U + eps) + eps)
#     U = torch.rand(shape)
#     g2 = -torch.log(-torch.log(U + eps) + eps)
#     return g1 - g2

# """Latent Generating Process"""
# B = torch.sigmoid((W + sample_gumbel(W.shape)) / tau) * ReLU_Y # B \in {0, 1}
# epsilon = exog_mean + torch.exp(exog_logvar / 2) * torch.randn(exog_mean.shape)
# latent = torch.sigmoid(torch.matmul(epsilon, torch.inverse(I - B)))

# inverse_sigmoid_latent = torch.log(latent / (1. - latent))

# # 1. with z
# xhat = decoder(latent)
# # 2. with Bz
# # xhat = decoder(torch.matmul(latent, B))
# # 3. with Bz + epsilon (SEM)
# # xhat = decoder(torch.matmul(latent, B) + epsilon)

# xhat = xhat.view(-1, 96, 96, 3)

# exog_mean, exog_logvar, inverse_sigmoid_latent, B, xhat
#%%