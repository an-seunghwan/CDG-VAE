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
        self.z_layer = nn.Linear(300, self.config["latent_dim"]) 
        self.logvar_layer = nn.Linear(300, 1) # 1 for diagonal variance (equal variance assumption)

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
                        requires_grad=True).to(device)
            )
        
        mask = np.triu(np.ones((self.config["latent_dim"], self.config["latent_dim"])), k=1)
        self.mask = torch.FloatTensor(mask).to(device)
        
        """decoder"""
        self.decoder = nn.Sequential(
            nn.Linear(self.config["latent_dim"], 300),
            nn.ELU(),
            nn.Linear(300, 300),
            nn.ELU(),
            nn.Linear(300, 3*96*96),
            nn.Tanh()
        ).to(device)
    
    def forward(self, input):
        h = self.encoder(nn.Flatten()(input.to(self.device)))
        z = self.z_layer(h)
        logvar = self.logvar_layer(h)
        
        Bz = torch.matmul(z, self.W * self.ReLU_Y * self.mask) # maksing is added
        epsilon = torch.randn(Bz.shape).to(self.device)
        z_sem = Bz + torch.exp(logvar / 2.) * epsilon
        
        xhat = self.decoder(z_sem)
        xhat = xhat.view(-1, 96, 96, 3)
        return z, logvar, Bz, z_sem, xhat
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
    z, logvar, Bz, z_sem, recon = model(batch)
    epsilon = torch.ones(Bz.shape)
    z_sem = Bz + torch.exp(logvar / 2.) * epsilon
    
    assert z.shape == (config["n"], config["latent_dim"])
    assert Bz.shape == (config["n"], config["latent_dim"])
    assert z_sem.shape == (config["n"], config["latent_dim"])
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
# x = torch.rand(4, 3, 96, 96)

# # encoder 
# encoder = []
# in_dim = 3
# for j in range(config["num_layer"]):
#     encoder.append(nn.Conv2d(in_channels=in_dim, out_channels=config["hidden_dim"] * (1 + j), kernel_size=4, stride=2))
#     encoder.append(nn.LeakyReLU(0.05))
#     in_dim = config["hidden_dim"] * (1 + j)
# encoder.append(nn.Flatten())
# encoder = nn.Sequential(*encoder)

# h = encoder(x)
# h.shape

# feature_layer = nn.Linear(40, config["latent_dim"])
# h = feature_layer(h)

# # weighted adjacency matrix
# p = {x:y for x,y in zip(np.arange(config["latent_dim"]), np.arange(config["latent_dim"]))}
# """build ReLU(Y)"""
# Y = torch.zeros((config["latent_dim"], config["latent_dim"]))
# for i in range(config["latent_dim"]):
#     for j in range(config["latent_dim"]):
#         Y[i, j] = p[j] - p[i]
# ReLU_Y = torch.nn.ReLU()(Y)

# W = torch.rand(config["latent_dim"], config["latent_dim"])
# min = -0.1
# max = 0.1
# W = (max - min) * W + min # ~ Uniform(-0.1, 0.1)
# W = W.fill_diagonal_(0.)
# W = nn.Parameter(W, requires_grad=True)

# B_trans_h = torch.matmul(h, W * ReLU_Y)

# class UnFlatten(nn.Module):
#     def forward(self, input, size=config["latent_dim"]):
#         return input.view(input.size(0), size, 1, 1)

# epsilon = torch.randn(B_trans_h.shape)
# h = UnFlatten()(B_trans_h + epsilon)

# # decoder
# decoder = []
# in_dim = config["latent_dim"]
# for j in reversed(range(1, config["num_layer"])):
#     decoder.append(nn.ConvTranspose2d(in_dim, config["hidden_dim"] * (1 + j), kernel_size=4, stride=2))
#     decoder.append(nn.LeakyReLU(0.05))
#     in_dim = config["hidden_dim"] * (1 + j)
# # decoder.append(nn.Flatten())
# decoder.append(nn.ConvTranspose2d(in_dim, 3, kernel_size=4, stride=2, padding=0))
# decoder.append(nn.ReflectionPad2d(1))
# decoder = nn.Sequential(*decoder)
# decoder(h).shape
#%%