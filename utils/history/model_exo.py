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
            nn.Linear(3*96*96, 900),
            nn.ELU(),
            nn.Linear(900, 300),
            nn.ELU(),
            nn.Linear(300, self.config["latent_dim"]),
        ).to(device)

        """weighted adjacency matrix"""
        p = {x:y for x,y in zip(range(config["latent_dim"]), range(config["latent_dim"]))}
        # build ReLU(Y)
        Y = torch.zeros((self.config["latent_dim"], self.config["latent_dim"]))
        for i in range(self.config["latent_dim"]):
            for j in range(self.config["latent_dim"]):
                Y[i, j] = p[j] - p[i]
        self.ReLU_Y = torch.nn.ReLU()(Y).to(device)

        self.W = nn.Parameter(
                torch.zeros((self.config["latent_dim"], self.config["latent_dim"]), 
                            requires_grad=True).to(self.device))
        # self.W = nn.Parameter(
        #     torch.nn.init.normal_(
        #         torch.zeros((self.config["latent_dim"], self.config["latent_dim"]), 
        #                     requires_grad=True),
        #         mean=0., std=0.1).to(self.device))
        
        # """masking"""
        # mask = np.triu(np.ones((self.config["latent_dim"], self.config["latent_dim"])), k=1)
        # self.mask = torch.FloatTensor(mask).to(self.device)
        
        """decoder"""
        self.decoder = nn.Sequential(
            nn.Linear(self.config["latent_dim"], 300),
            nn.ELU(),
            nn.Linear(300, 300),
            nn.ELU(),
            nn.Linear(300, 3*96*96),
            nn.Tanh()
        ).to(device)
        
        self.I = torch.eye(self.config["latent_dim"]).to(device)
        
    def forward(self, input):
        # h = self.encoder(nn.Flatten()(input.to(self.device)))
        # exog_mean, exog_logvar = torch.split(h, self.config["latent_dim"], dim=1)
        # exog_logvar = torch.tanh(exog_logvar) * 0.5 # variance scaling (exp(-0.5) ~ exp(0.5))
        exog_mean = self.encoder(nn.Flatten()(input.to(self.device)))

        """Latent Generating Process"""
        # epsilon = exog_mean + torch.exp(exog_logvar / 2) * torch.randn(exog_mean.shape).to(self.device)
        epsilon = exog_mean + torch.randn(exog_mean.shape).to(self.device)
        B = self.W * self.ReLU_Y 
        # B = self.W * self.ReLU_Y * self.mask # masking
        latent = torch.matmul(epsilon, torch.inverse(self.I - B))
        
        # 1. with z
        # xhat = self.decoder(latent)
        # 2. with Bz
        # xhat = self.decoder(torch.matmul(latent, B))
        # 3. with Bz + epsilon (SEM)
        xhat = self.decoder(torch.matmul(latent, B) + epsilon)
        
        # using z = Bz + epsilon relationship
        # xhat2 = self.decoder(latent)
        
        # xhat1 = xhat1.view(-1, 96, 96, 3)
        # xhat2 = xhat2.view(-1, 96, 96, 3) 
        xhat = xhat.view(-1, 96, 96, 3)
        # return exog_mean, exog_logvar, latent, B, xhat1, xhat2
        return exog_mean, latent, B, xhat
#%%
def main():
    config = {
        "n": 100,
        "latent_dim": 5,
    }
    
    model = VAE(config, 'cpu')
    for x in model.parameters():
        print(x.shape)
        
    batch = torch.rand(config["n"], 96, 96, 3)
    latent, logvar, zB, B, xhat = model(batch)
    
    assert latent.shape == (config["n"], config["latent_dim"])
    assert logvar.shape == (config["n"], 1)
    assert zB.shape == (config["n"], config["latent_dim"])
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
#     nn.Linear(3*96*96, 900),
#     nn.ELU(),
#     nn.Linear(900, 300),
#     nn.ELU(),
# )
# z_layer = nn.Linear(300, config["latent_dim"]) 
# logvar_layer = nn.Linear(300, 1) # 1 for diagonal variance (equal variance assumption)

# """weighted adjacency matrix"""
# p = {x:y for x,y in zip(range(config["latent_dim"]), range(config["latent_dim"]))}
# # build ReLU(Y)
# Y = torch.zeros((config["latent_dim"], config["latent_dim"]))
# for i in range(config["latent_dim"]):
#     for j in range(config["latent_dim"]):
#         Y[i, j] = p[j] - p[i]
# ReLU_Y = torch.nn.ReLU()(Y)

# W = nn.Parameter(
#         torch.zeros((config["latent_dim"], config["latent_dim"]), 
#                     requires_grad=True))

# # mask = np.triu(np.ones((config["latent_dim"], config["latent_dim"])), k=1)
# # mask = torch.FloatTensor(mask)

# x = torch.randn(10, 96, 96, 3)
# h = encoder(nn.Flatten()(x))
# z = z_layer(h)
# logvar = logvar_layer(h)

# latent = torch.zeros(z.shape)
# for j in range(config["latent_dim"]):
#     if j == 0:
#         latent[:, j] = z[:, j].clone()
#     latent[:, j] = latent[:, j-1].clone() + torch.abs(z[:, j].clone())

# # B = W * ReLU_Y * mask
# B = W * ReLU_Y 
# zB = torch.matmul(latent, B)
# epsilon = torch.randn(z.shape)
# z = zB + torch.exp(logvar / 2) * epsilon

# """decoder"""
# decoder = nn.Sequential(
#     nn.Linear(config["latent_dim"], 300),
#     nn.ELU(),
#     nn.Linear(300, 300),
#     nn.ELU(),
#     nn.Linear(300, 3*96*96),
#     nn.Tanh()
# )

# xhat = decoder(z)
# xhat = xhat.view(-1, 96, 96, 3)
#%%