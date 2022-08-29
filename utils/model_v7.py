#%%
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
#%%
class PlanarFlows(nn.Module):
    """invertible transformation with tanh
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
                z = h - u_.t() * torch.tanh(z @ self.w[j] + self.b[j])
            h = z
        return h
            
    def forward(self, inputs):
        h = inputs
        for j in range(self.flow_num):
            u_ = self.build_u(self.u[j], self.w[j])
            h = h + u_.t() * torch.tanh(h @ self.w[j] + self.b[j])
        return h
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
        ).to(device)
        
        self.B = B.to(device) # binary adjacency matrix
        
        """Generalized Linear SEM: Invertible NN"""
        self.flows = [PlanarFlows(config["node_dim"], config["flow_num"], config["inverse_loop"], device) 
                    for _ in range(config["node"])]
        
        """decoder"""
        self.decoder = nn.Sequential(
            nn.Linear(config["node"] * config["node_dim"], 300),
            nn.ELU(),
            nn.Linear(300, 300),
            nn.ELU(),
            nn.Linear(300, 3*96*96),
            nn.Tanh()
        ).to(device)
        
        """Prior"""
        self.prior = [PlanarFlows(config["node_dim"], config["flow_num"], config["inverse_loop"], device) 
                    for _ in range(config["node"])]
        
        # """Align"""
        # self.align = [nn.BatchNorm1d(config["node_dim"]).to(device) for _ in range(config["node"])]
        
        self.I = torch.eye(config["node"]).to(device)

    def inverse(self, input):
        input = torch.split(input, self.config["node_dim"], dim=1)
        inverse_latent = list(map(lambda x, layer: layer.inverse(x), input, self.flows))
        return inverse_latent
    
    def encode(self, input, withnoise=True):
        image = input
        h = self.encoder(nn.Flatten()(image)) # [batch, node * node_dim * 2]
        mean, logvar = torch.split(h, self.config["node"] * self.config["node_dim"], dim=1)
        
        if withnoise:
            noise = torch.randn(image.size(0), self.config["node"] * self.config["node_dim"]).to(self.device) 
            epsilon = mean + torch.exp(logvar / 2) * noise
            epsilon = epsilon.view(-1, self.config["node_dim"], self.config["node"]).contiguous()
        else:
            epsilon = mean
            epsilon = epsilon.view(-1, self.config["node_dim"], self.config["node"]).contiguous()
        latent = torch.matmul(epsilon, torch.inverse(self.I - self.B)) # [batch, node_dim, node]
        latent_orig = latent.clone()
        latent = [x.squeeze(dim=2) for x in torch.split(latent, 1, dim=2)] # [batch, node_dim] x node
        causal_latent = list(map(lambda x, layer: layer(x), latent, self.flows))
        causal_latent = torch.cat(causal_latent, dim=1)
        
        return mean, logvar, latent_orig, causal_latent
    
    def decode(self, input):
        xhat = self.decoder(input).view(-1, 96, 96, 3)
        return xhat
    
    def forward(self, input):
        image, label = input
        h = self.encoder(nn.Flatten()(image)) # [batch, node * node_dim * 2]
        mean, logvar = torch.split(h, self.config["node"] * self.config["node_dim"], dim=1)
        
        """Latent Generating Process"""
        noise = torch.randn(image.size(0), self.config["node"] * self.config["node_dim"]).to(self.device) 
        epsilon = mean + torch.exp(logvar / 2) * noise
        epsilon = epsilon.view(-1, self.config["node_dim"], self.config["node"]).contiguous()
        latent = torch.matmul(epsilon, torch.inverse(self.I - self.B)) # [batch, node_dim, node]
        latent_orig = latent.clone()
        latent = [x.squeeze(dim=2) for x in torch.split(latent, 1, dim=2)] # [batch, node_dim] x node
        causal_latent = list(map(lambda x, layer: layer(x), latent, self.flows))
        causal_latent = torch.cat(causal_latent, dim=1)
        
        xhat = self.decoder(causal_latent)
        xhat = xhat.view(-1, 96, 96, 3)
        
        """prior"""
        label_ = torch.repeat_interleave(label, self.config["node_dim"], dim=1)
        prior_logvar = list(map(lambda x, layer: torch.transpose(layer(x).unsqueeze(dim=1), 1, 2), 
                        torch.split(label_, self.config["node_dim"], dim=1), self.prior))
        prior_logvar = torch.cat(prior_logvar, dim=2).view(-1, self.config["node_dim"] * self.config["node"]).contiguous()
        
        # """Alignment"""
        # align_latent = list(map(lambda x, layer: layer(x), causal_latent, self.align))
        
        return mean, logvar, prior_logvar, latent_orig, causal_latent, xhat
#%%
def main():
    config = {
        "n": 10,
        "node": 4,
        "node_dim": 2, 
        "flow_num": 4,
        "inverse_loop": 100,
        "prior_flow_num": 8,
    }
    
    B = torch.zeros(config["node"], config["node"])
    B[:2, 2:] = 1
    model = VAE(B, config, 'cpu')
    for x in model.parameters():
        print(x.shape)
        
    batch = torch.rand(config["n"], 96, 96, 3)
    u = torch.rand(config["n"], config["node"])
    mean, logvar, prior_logvar, latent_orig, causal_latent, xhat = model([batch, u])
    
    assert mean.shape == (config["n"], config["node"] * config["node_dim"])
    assert logvar.shape == (config["n"], config["node"] * config["node_dim"])
    assert prior_logvar.shape == (config["n"], config["node"] * config["node_dim"])
    assert latent_orig.shape == (config["n"], config["node_dim"], config["node"])
    assert causal_latent.shape == (config["n"], config["node_dim"] * config["node"])
    # assert align_latent[0].shape == (config["n"], config["node_dim"])
    # assert len(align_latent) == config["node"]
    assert xhat.shape == (config["n"], 96, 96, 3)
    
    inverse_diff = torch.abs(sum([x - y for x, y in zip([x.squeeze(dim=2) for x in torch.split(latent_orig, 1, dim=2)], 
                                                        model.inverse(causal_latent))]).sum())
    assert inverse_diff / (config["n"] * config["node"] * config["node_dim"]) < 1e-5
    
    print("Model test pass!")
#%%
if __name__ == '__main__':
    main()
#%%
# config = {
#     "n": 10,
#     "node": 4,
#     "node_dim": 2, 
#     "flow_num": 4,
#     "inverse_loop": 100,
#     "prior_flow_num": 8,
# }

# device = 'cpu'
# B = torch.zeros(config["node"], config["node"])
# B[:2, 2:] = 1
    

# """encoder"""
# encoder = nn.Sequential(
#     nn.Linear(3*96*96, 300),
#     nn.ELU(),
#     nn.Linear(300, 300),
#     nn.ELU(),
#     nn.Linear(300, config["node"] * config["node_dim"] * 2),
# ).to(device)

# B = B.to(device) # binary adjacency matrix

# """Generalized Linear SEM: Invertible NN"""
# flows = [PlanarFlows(config["node_dim"], config["flow_num"], config["inverse_loop"], device) 
#          for _ in range(config["node"])]

# """decoder"""
# decoder = nn.Sequential(
#     nn.Linear(config["node"] * config["node_dim"], 300),
#     nn.ELU(),
#     nn.Linear(300, 300),
#     nn.ELU(),
#     nn.Linear(300, 3*96*96),
#     nn.Tanh()
# ).to(device)

# """Prior"""
# prior = [PlanarFlows(config["node_dim"], config["flow_num"], config["inverse_loop"], device) 
#          for _ in range(config["node"])]

# # """Align"""
# # align = [nn.BatchNorm1d(config["node_dim"]).to(device) for _ in range(config["node"])]

# I = torch.eye(config["node"]).to(device)
# #%%
# w = [(nn.Parameter(torch.randn(config["node_dim"], 1) * 0.1).to(device))
#             for _ in range(config["flow_num"])]
# b = [nn.Parameter((torch.randn(1, 1) * 0.1).to(device))
#             for _ in range(config["flow_num"])]
# u = [nn.Parameter((torch.randn(config["node_dim"], 1) * 0.1).to(device))
#             for _ in range(config["flow_num"])]

# def build_u(u_, w_):
#     """sufficient condition to be invertible"""
#     term1 = -1 + torch.log(1 + torch.exp(w_.t() @ u_))
#     term2 = w_.t() @ u_
#     u_hat = u_ + (term1 - term2) * (w_ / torch.norm(w_, p=2) ** 2)
#     return u_hat
# #%%
# image = torch.randn(10, 96, 96, 3)
# u = torch.randn(10, config["node"])

# h = encoder(nn.Flatten()(image)) # [batch, node * node_dim * 2]
# mean, logvar = torch.split(h, config["node"] * config["node_dim"], dim=1)

# """Latent Generating Process"""
# noise = torch.randn(image.size(0), config["node"] * config["node_dim"]).to(device) 
# epsilon = mean + torch.exp(logvar / 2) * noise
# epsilon = epsilon.view(-1, config["node_dim"], config["node"]).contiguous()
# latent = torch.matmul(epsilon, torch.inverse(I - B)) # [batch, node_dim, node]
# latent_orig = latent.clone()
# latent = [x.squeeze(dim=2) for x in torch.split(latent, 1, dim=2)] # [batch, node_dim] x node
# causal_latent = list(map(lambda x, layer: layer(x), latent, flows))
# causal_latent = torch.cat(causal_latent, dim=1)

# xhat = decoder(causal_latent)
# xhat = xhat.view(-1, 96, 96, 3)

# """prior"""
# u = torch.repeat_interleave(u, 2, dim=1)
# prior_logvar = list(map(lambda x, layer: torch.transpose(layer(x).unsqueeze(dim=1), 1, 2), 
#                         torch.split(u, config["node_dim"], dim=1), prior))
# prior_logvar = torch.cat(prior_logvar, dim=2).view(-1, config["node_dim"] * config["node"]).contiguous()

# # """Alignment"""
# # align_latent = list(map(lambda x, layer: layer(x), causal_latent, align))
#%%