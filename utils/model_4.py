#%%
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
#%%
class RaidalFlows(nn.Module):
    """invertible transformation
    Reference:
    [1]: http://proceedings.mlr.press/v37/rezende15.pdf
    [2]: https://arxiv.org/pdf/1811.00995.pdf
    """
    def __init__(self,
                 input_dim,
                 flow_num,
                 inverse_loop,
                 device='cpu'):
        super(RaidalFlows, self).__init__()

        self.input_dim = input_dim
        self.flow_num = flow_num
        self.inverse_loop = inverse_loop
        
        self.z0 = [nn.Parameter(torch.randn(1, self.input_dim)).to(device)
                  for _ in range(self.flow_num)]
        self.alpha = [nn.Parameter(torch.randn(1, 1).to(device))
                    for _ in range(self.flow_num)]
        self.beta = [nn.Parameter(torch.randn(1, 1).to(device))
                    for _ in range(self.flow_num)]
        
    def build_beta(self, alpha_, beta_):
        """sufficient condition to be invertible"""
        m = -1 + torch.log(1 + torch.exp(beta_))
        beta_hat = -alpha_ + m
        return beta_hat
    
    def inverse(self, inputs):
        h = inputs
        for j in reversed(range(self.flow_num)):
            z = h
            for _ in range(self.inverse_loop):
                beta_ = self.build_beta(self.alpha[j], self.beta[j])
                r = torch.norm(z - self.z0[j], p=2)
                z = h - beta_ / (self.alpha[j] + r) * (z - self.z0[j])
            h = z
        return h
            
    def forward(self, inputs):
        h = inputs
        for j in range(self.flow_num):
            beta_ = self.build_beta(self.alpha[j], self.beta[j])
            r = torch.norm(h - self.z0[j], p=2)
            h = h + beta_ / (self.alpha[j] + r) * (h - self.z0[j])
        return h
#%%
class VAE(nn.Module):
    def __init__(self, mask, config, device):
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
        self.mask = mask.to(device) 
        self.W = nn.Parameter(
            torch.randn((self.config["node"], self.config["node"]), 
                        requires_grad=True).to(device)
            )
        self.I = torch.eye(config["node"]).to(device)
        
        """Generalized Linear SEM: Invertible NN"""
        self.flows = [RaidalFlows(config["node_dim"], config["flow_num"], config["inverse_loop"], device) 
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
    
    def forward(self, input):
        h = self.encoder(nn.Flatten()(input)) # [batch, node * node_dim * 2]
        mean, logvar = torch.split(h, self.config["node"] * self.config["node_dim"], dim=1)
        
        B = self.W * self.mask
        I_B_inv = torch.inverse(self.I - B)
        
        """Latent Generating Process"""
        noise = torch.randn(input.size(0), self.config["node"] * self.config["node_dim"]).to(self.device) 
        epsilon = mean + torch.exp(logvar / 2) * noise
        epsilon = epsilon.view(-1, self.config["node_dim"], self.config["node"]).contiguous()
        latent = torch.matmul(epsilon, I_B_inv) # [batch, node_dim, node]
        orig_latent = latent.clone()
        latent = [x.squeeze(dim=2) for x in torch.split(latent, 1, dim=2)] # [batch, node_dim] x node
        latent = list(map(lambda x, layer: layer(x), latent, self.flows)) # [batch, node_dim] x node
        
        xhat = self.decoder(torch.cat(latent, dim=1))
        xhat = xhat.view(-1, self.config["image_size"], self.config["image_size"], 3)
        
        """Alignment"""
        mean_ = mean.view(-1, self.config["node_dim"], self.config["node"]).contiguous() # deterministic part
        align_latent = torch.matmul(mean_, I_B_inv) # [batch, node_dim, node]
        align_latent = [x.squeeze(dim=2) for x in torch.split(align_latent, 1, dim=2)] # [batch, node_dim] x node
        align_latent = list(map(lambda x, layer: layer(x), align_latent, self.flows)) # [batch, node_dim] x node
        
        return mean, logvar, orig_latent, latent, align_latent, xhat
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
    mean, logvar, orig_latent, latent, align_latent, xhat = model(batch)
    
    assert mean.shape == (config["n"], config["node"] * config["node_dim"])
    assert logvar.shape == (config["n"], config["node"] * config["node_dim"])
    assert orig_latent.shape == (config["n"], config["node_dim"], config["node"])
    assert latent[0].shape == (config["n"], config["node_dim"])
    assert len(latent) == config["node"]
    assert align_latent[0].shape == (config["n"], config["node_dim"])
    assert len(align_latent) == config["node"]
    assert xhat.shape == (config["n"], config["image_size"], config["image_size"], 3)
    
    mean, logvar, orig_latent, latent, align_latent, xhat = model(batch)
    
    inverse_diff = torch.abs(sum([x - y for x, y in zip([x.squeeze(dim=2) for x in torch.split(orig_latent, 1, dim=2)], 
                                                        model.inverse(latent))]).sum())
    assert inverse_diff / (config["n"] * config["node"] * config["node_dim"]) < 1e-5
    
    print("Model test pass!")
#%%
if __name__ == '__main__':
    main()
#%%
# device = 'cpu'
# input_dim = 2
# flow_num = 4
# inverse_loop = 100

# z0 = [nn.Parameter(torch.randn(1, input_dim)).to(device)
#             for _ in range(flow_num)]
# alpha = [nn.Parameter(torch.randn(1, 1).to(device))
#             for _ in range(flow_num)]
# beta = [nn.Parameter(torch.randn(1, 1).to(device))
#             for _ in range(flow_num)]
    
# def build_beta(alpha_, beta_):
#     """sufficient condition to be invertible"""
#     m = -1 + torch.log(1 + torch.exp(beta_))
#     beta_hat = -alpha_ + m
#     return beta_hat

# def inverse(self, inputs):
#     h = inputs
#     for j in reversed(range(self.flow_num)):
#         z = h
#         for _ in range(self.inverse_loop):
#             beta_ = self.build_beta(self.alpha[j], self.beta[j])
#             r = torch.norm(z - self.z0[j], p=2)
#             z = h - beta_ / (self.alpha[j] + r) * (z - self.z0[j])
#         h = z
#     return h
        
# h = torch.randn(10, 2)
# for j in range(flow_num):
#     beta_ = build_beta(alpha[j], beta[j])
#     r = torch.norm(h - z0[j], p=2)
#     h = h + beta_ / (alpha[j] + r) * (h - z0[j])
#%%