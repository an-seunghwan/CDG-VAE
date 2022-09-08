#%%
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
#%%
class PlanarFlows(nn.Module):
    """invertible transformation with ELU
    ELU: h(x) = 
    x if x > 0
    alpha * (exp(x) - 1) if x <= 0
    
    gradient of h(x) = 
    1 if x > 0
    alpha * exp(x) if x <= 0
    
    if alpha <= 1, then 0 < gradient of h(x) <=1 
    
    Reference:
    [1]: http://proceedings.mlr.press/v37/rezende15.pdf
    [2]: https://arxiv.org/pdf/1811.00995.pdf
    """
    def __init__(self,
                 input_dim,
                 flow_num,
                 inverse_loop,
                 device='cpu'):
        super(PlanarFlows, self).__init__()

        self.input_dim = input_dim
        self.flow_num = flow_num
        self.inverse_loop = inverse_loop
        self.alpha = torch.tensor(1, dtype=torch.float32) # parameter of ELU
        
        self.w = [(nn.Parameter(torch.randn(self.input_dim, 1) * 0.1).to(device))
                  for _ in range(self.flow_num)]
        self.b = [nn.Parameter((torch.randn(1, 1) * 0.1).to(device))
                  for _ in range(self.flow_num)]
        self.u = [nn.Parameter((torch.randn(self.input_dim, 1) * 0.1).to(device))
                  for _ in range(self.flow_num)]
        
    def build_u(self, u_, w_):
        """sufficient condition to be invertible"""
        term1 = -1 + torch.log(1 + torch.exp(w_.t() @ u_))
        term2 = w_.t() @ u_
        u_hat = u_ + (term1 - term2) * (w_ / torch.norm(w_, p=2) ** 2)
        return u_hat
    
    def inverse(self, inputs):
        h = inputs
        for j in reversed(range(self.flow_num)):
            z = h
            for _ in range(self.inverse_loop):
                u_ = self.build_u(self.u[j], self.w[j]) 
                z = h - u_.t() * F.elu(z @ self.w[j] + self.b[j], alpha=self.alpha)
            h = z
        return h
            
    def forward(self, inputs, log_determinant=False):
        h = inputs
        logdet = 0
        for j in range(self.flow_num):
            u_ = self.build_u(self.u[j], self.w[j])
            if log_determinant:
                x = h @ self.w[j] + self.b[j]
                gradient = torch.where(x > 0, 
                                       torch.tensor(1, dtype=torch.float32), 
                                       self.alpha * torch.exp(x))
                psi = gradient * self.w[j].squeeze()
                logdet += torch.log((1 + psi @ u_).abs())
            h = h + u_.t() * F.elu(h @ self.w[j] + self.b[j], alpha=self.alpha)
        return h, logdet
#%%
class VAE(nn.Module):
    def __init__(self, B, config, device):
        super(VAE, self).__init__()
        
        self.config = config
        self.device = device
        
        """encoder"""
        self.encoder = nn.Sequential(
            nn.Linear(3*config["image_size"]*config["image_size"], 300),
            nn.ELU(),
            nn.Linear(300, 300),
            nn.ELU(),
            nn.Linear(300, config["node"] * config["node_dim"] * 2),
        ).to(device)
        
        """Causal Adjacency Matrix"""
        self.B = B.to(device) 
        self.I = torch.eye(config["node"]).to(device)
        self.I_B_inv = torch.inverse(self.I - self.B)
        
        """Generalized Linear SEM: Invertible NN"""
        self.flows = [PlanarFlows(config["node_dim"], config["flow_num"], config["inverse_loop"], device) 
                    for _ in range(config["node"])]
        
        """decoder"""
        self.decoder = nn.Sequential(
            nn.Linear(config["node"] * config["node_dim"], 300),
            nn.ELU(),
            nn.Linear(300, 300),
            nn.ELU(),
            nn.Linear(300, 3*config["image_size"]*config["image_size"]),
            nn.Tanh()
        ).to(device)
        
    def inverse(self, input): 
        inverse_latent = list(map(lambda x, layer: layer.inverse(x), input, self.flows))
        return inverse_latent
    
    def get_posterior(self, input):
        h = self.encoder(nn.Flatten()(input)) # [batch, node * node_dim * 2]
        mean, logvar = torch.split(h, self.config["node"] * self.config["node_dim"], dim=1)
        return mean, logvar
    
    def transform(self, input, log_determinant=False):
        latent = torch.matmul(input, self.I_B_inv) # [batch, node_dim, node], input = epsilon (exogenous variables)
        orig_latent = latent.clone()
        latent = [x.squeeze(dim=2) for x in torch.split(latent, 1, dim=2)] # [batch, node_dim] x node
        latent = list(map(lambda x, layer: layer(x, log_determinant=log_determinant), latent, self.flows)) # input = (I-B^T)^{-1} * epsilon
        logdet = [x[1] for x in latent]
        latent = [x[0] for x in latent]
        return orig_latent, latent, logdet
    
    def encode(self, input, deterministic=False, log_determinant=False):
        mean, logvar = self.get_posterior(input)
        
        """Latent Generating Process"""
        if deterministic:
            epsilon = mean
        else:
            noise = torch.randn(input.size(0), self.config["node"] * self.config["node_dim"]).to(self.device) 
            epsilon = mean + torch.exp(logvar / 2) * noise
        epsilon = epsilon.view(-1, self.config["node_dim"], self.config["node"]).contiguous()
        orig_latent, latent, logdet = self.transform(epsilon, log_determinant=log_determinant)
        
        return mean, logvar, epsilon, orig_latent, latent, logdet
    
    def forward(self, input, deterministic=False, log_determinant=False):
        """encoding"""
        mean, logvar, epsilon, orig_latent, latent, logdet = self.encode(input, 
                                                                         deterministic=deterministic,
                                                                         log_determinant=log_determinant)
        
        """decoding"""
        xhat = self.decoder(torch.cat(latent, dim=1))
        xhat = xhat.view(-1, self.config["image_size"], self.config["image_size"], 3)
        
        """Alignment"""
        _, _, _, _, align_latent, _ = self.encode(input, 
                                                  deterministic=True, 
                                                  log_determinant=log_determinant)
        
        return mean, logvar, epsilon, orig_latent, latent, logdet, align_latent, xhat
#%%
def main():
    config = {
        "image_size": 64,
        "n": 64,
        "node": 4,
        "node_dim": 2, 
        "flow_num": 4,
        "inverse_loop": 100,
    }
    
    B = torch.zeros(config["node"], config["node"])
    B[:2, 2:] = 1
    model = VAE(B, config, 'cpu')
    for x in model.parameters():
        print(x.shape)
        
    batch = torch.rand(config["n"], config["image_size"], config["image_size"], 3)
    mean, logvar, epsilon, orig_latent, latent, logdet, align_latent, xhat = model(batch)
    mean, logvar, epsilon, orig_latent, latent, logdet, align_latent, xhat = model(batch, log_determinant=True)
    
    assert mean.shape == (config["n"], config["node"] * config["node_dim"])
    assert logvar.shape == (config["n"], config["node"] * config["node_dim"])
    assert epsilon.shape == (config["n"], config["node_dim"], config["node"])
    assert orig_latent.shape == (config["n"], config["node_dim"], config["node"])
    assert latent[0].shape == (config["n"], config["node_dim"])
    assert len(latent) == config["node"]
    assert logdet[0].shape == (config["n"], 1)
    assert len(logdet) == config["node"]
    assert align_latent[0].shape == (config["n"], config["node_dim"])
    assert len(align_latent) == config["node"]
    assert xhat.shape == (config["n"], config["image_size"], config["image_size"], 3)
    
    # deterministic behavior
    out1 = model(batch, deterministic=False)
    out2 = model(batch, deterministic=True)
    assert (out1[0] - out2[0]).abs().mean() == 0 # mean
    assert (out1[1] - out2[1]).abs().mean() == 0 # logvar
    assert (torch.cat(out1[4], dim=1) - torch.cat(out2[4], dim=1)).abs().mean() != 0 # latent
    assert (torch.cat(out1[6], dim=1) - torch.cat(out2[6], dim=1)).abs().mean() == 0 # align_latent
    
    inverse_diff = torch.abs(sum([x - y for x, y in zip([x.squeeze(dim=2) for x in torch.split(orig_latent, 1, dim=2)], 
                                                        model.inverse(latent))]).sum())
    assert inverse_diff / (config["n"] * config["node"] * config["node_dim"]) < 1e-5
    
    print("Model test pass!")
#%%
if __name__ == '__main__':
    main()
#%%
# input_dim = 2
# flow_num = 4
# inverse_loop = 100
# device = 'cpu'
# log_determinant = True
# alpha = torch.tensor(1, dtype=torch.float32) # parameter of ELU

# w = [(nn.Parameter(torch.randn(input_dim, 1) * 0.1).to(device))
#             for _ in range(flow_num)]
# b = [nn.Parameter((torch.randn(1, 1) * 0.1).to(device))
#             for _ in range(flow_num)]
# u = [nn.Parameter((torch.randn(input_dim, 1) * 0.1).to(device))
#             for _ in range(flow_num)]

# def build_u(u_, w_):
#     """sufficient condition to be invertible"""
#     term1 = -1 + torch.log(1 + torch.exp(w_.t() @ u_))
#     term2 = w_.t() @ u_
#     u_hat = u_ + (term1 - term2) * (w_ / torch.norm(w_, p=2) ** 2)
#     return u_hat

# rand = torch.randn(10, input_dim)
# h = rand
# logdet = 0
# for j in range(flow_num):
#     u_ = build_u(u[j], w[j])
#     if log_determinant:
#         x = h @ w[j] + b[j]
#         gradient = torch.where(x > 0, torch.tensor(1, dtype=torch.float32), alpha * torch.exp(x))
#         psi = gradient * w[j].squeeze()
#         logdet += torch.log((1 + psi @ u_).abs())
#     h = h + u_.t() * F.elu(h @ w[j] + b[j], alpha=alpha)
# #%%
# for j in reversed(range(flow_num)):
#     z = h
#     for _ in range(inverse_loop):
#         u_ = build_u(u[j], w[j]) 
#         z = h - u_.t() * F.elu(z @ w[j] + b[j], alpha=alpha)
#     h = z
# h
#%%
# config = {
#     "image_size": 64,
#     "n": 64,
#     "node": 4,
#     "node_dim": 2, 
#     "flow_num": 4,
#     "inverse_loop": 100,
# }

# device = 'cpu'
# B = torch.zeros(config["node"], config["node"])
# B[:2, 2:] = 1
# batch = torch.rand(config["n"], config["image_size"], config["image_size"], 3)
    
# """encoder"""
# encoder = nn.Sequential(
#     nn.Linear(3*config["image_size"]*config["image_size"], 300),
#     nn.ELU(),
#     nn.Linear(300, 300),
#     nn.ELU(),
#     nn.Linear(300, config["node"] * config["node_dim"] * 2),
# ).to(device)

# """Causal Adjacency Matrix"""
# B = B.to(device) 
# I = torch.eye(config["node"]).to(device)
# I_B_inv = torch.inverse(I - B)

# """Generalized Linear SEM: Invertible NN"""
# flows = [PlanarFlows(config["node_dim"], config["flow_num"], config["inverse_loop"], device) 
#             for _ in range(config["node"])]

# """decoder"""
# decoder = nn.Sequential(
#     nn.Linear(config["node"] * config["node_dim"], 300),
#     nn.ELU(),
#     nn.Linear(300, 300),
#     nn.ELU(),
#     nn.Linear(300, 3*config["image_size"]*config["image_size"]),
#     nn.Tanh()
# ).to(device)
# #%%
# def inverse(self, input): 
#     inverse_latent = list(map(lambda x, layer: layer.inverse(x), input, flows))
#     return inverse_latent
# #%%
# # def encode(self, input, deterministic=False):
# deterministic=False
# log_determinant=True

# input = batch
# h = encoder(nn.Flatten()(input)) # [batch, node * node_dim * 2]
# mean, logvar = torch.split(h, config["node"] * config["node_dim"], dim=1)

# """Latent Generating Process"""
# if deterministic:
#     epsilon = mean
# else:
#     noise = torch.randn(input.size(0), config["node"] * config["node_dim"]).to(device) 
#     epsilon = mean + torch.exp(logvar / 2) * noise
# epsilon = epsilon.view(-1, config["node_dim"], config["node"]).contiguous()
# latent = torch.matmul(epsilon, I_B_inv) # [batch, node_dim, node]
# orig_latent = latent.clone()
# latent = [x.squeeze(dim=2) for x in torch.split(latent, 1, dim=2)] # [batch, node_dim] x node
# latent = list(map(lambda x, layer: layer(x, log_determinant=log_determinant), latent, flows)) # [batch, node_dim] x node
# logdet = [x[1] for x in latent]
# latent = [x[0] for x in latent]
#%%