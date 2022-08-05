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
        ).to(device)
        self.logvar_layer = nn.Linear(300, config["latent_dim"]).to(device) # 1 for single diagonal variance (equal variance assumption)

        """weighted adjacency matrix"""
        p = {x:y for x,y in zip(range(config["latent_dim"]), range(config["latent_dim"]))}
        # build ReLU(Y)
        Y = torch.zeros((self.config["latent_dim"], self.config["latent_dim"]))
        for i in range(self.config["latent_dim"]):
            for j in range(self.config["latent_dim"]):
                Y[i, j] = p[j] - p[i]
        self.ReLU_Y = torch.nn.ReLU()(Y).to(device)

        self.W = nn.Parameter(
            torch.nn.init.normal_(
                torch.zeros((self.config["latent_dim"], self.config["latent_dim"]), 
                            requires_grad=True),
                mean=0., std=0.1).to(self.device))
        
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
        h = self.encoder(nn.Flatten()(input.to(self.device)))
        logvar = self.logvar_layer(h)
        
        z = torch.zeros(logvar.shape[0], self.config["latent_dim"]).to(self.device)
        Bz = torch.zeros(logvar.shape[0], self.config["latent_dim"]).to(self.device) # posterior mean vector
        B = self.W * self.ReLU_Y
        
        """Latent Generating Process"""
        epsilon = torch.exp(logvar / 2.) * torch.randn(z.shape).to(self.device)
        for j in range(self.config["latent_dim"]):
            if j == 0:  # root node
                z[:, [j]] = epsilon[:, [j]]
            Bz[:, [j]] = torch.matmul(z[:, :j].clone(), B[:j, [j]].clone()) # posterior mean 
            z[:, [j]] = Bz[:, [j]].clone() + epsilon[:, [j]]
        
        xhat = self.decoder(z)
        xhat = xhat.view(-1, 96, 96, 3)
        return z, logvar, Bz, B, xhat
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
    z, logvar, recon = model(batch)
    
    assert z.shape == (config["n"], config["latent_dim"])
    assert logvar.shape == (config["n"], 1)
    assert recon.shape == (config["n"], 96, 96, 3)
    
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
# # z_layer = nn.Linear(300, config["latent_dim"]) 
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
#     torch.nn.init.normal_(
#         torch.zeros((config["latent_dim"], config["latent_dim"]), 
#                     requires_grad=True),
#         mean=0., std=1.))

# # mask = np.triu(np.ones((config["latent_dim"], config["latent_dim"])), k=1)
# # mask = torch.FloatTensor(mask)

# x = torch.randn(10, 96, 96, 3)
# h = encoder(nn.Flatten()(x))
# # z = z_layer(h)
# logvar = logvar_layer(h)

# """Latent Generating Process"""
# z = torch.zeros(logvar.shape[0], config["latent_dim"])
# Bz = torch.zeros(logvar.shape[0], config["latent_dim"])
# B = W * ReLU_Y
# for j in range(config["latent_dim"]):
#     epsilon = torch.randn(logvar.shape)
#     if j == 0: 
#         z[:, [j]] = torch.exp(logvar / 2.) * epsilon
#     Bz[:, [j]] = torch.matmul(z[:, :j], B[:j, [j]])
#     z[:, [j]] = torch.matmul(z[:, :j], B[:j, [j]]) + torch.exp(logvar / 2.) * epsilon

# torch.matmul(z, B)

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


# def forward(self, input):
# h = encoder(nn.Flatten()(input.to(device)))
# z = z_layer(h)
# logvar = logvar_layer(h)

# Bz = torch.matmul(z, W * ReLU_Y * mask) # maksing is added
# epsilon = torch.randn(Bz.shape).to(device)
# z_sem = Bz + torch.exp(logvar / 2.) * epsilon

# xhat = self.decoder(z_sem)
# xhat = xhat.view(-1, 96, 96, 3)
# return z, logvar, Bz, z_sem, xhat
#%%