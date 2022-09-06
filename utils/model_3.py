#%%
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
#%%
class InvertiblePriorLinear(nn.Module):
    """Invertible Prior for Linear case

    Parameter:
        p: mean and std parameter for scaling
    """
    def __init__(self, device='cpu'):
        super(InvertiblePriorLinear, self).__init__()
        self.p = nn.Parameter(torch.rand([2])).to(device)

    def forward(self, eps):
        o = self.p[0] * eps + self.p[1]
        return o
    
    def inverse(self, o):
        eps = (o - self.p[1]) / self.p[0]
        return eps
#%%
class InvertiblePWL(nn.Module):
    """_summary_
    Reference:
    [1]: https://github.com/xwshen51/DEAR/blob/main/causal_model.py

    Args:
        nn (_type_): _description_
    """
    def __init__(self, vmin=-5, vmax=5, n=100, use_bias=True, device='cpu'):
        super(InvertiblePWL, self).__init__()
        
        self.int_length = (vmax - vmin) / (n - 1)
        self.vmin = vmin
        self.vmax = vmax
        self.n = n
        self.device = device
        if use_bias:
            self.b = nn.Parameter(torch.randn([1]).to(device) + vmin)
        else:
            self.b = vmin
        self.points = nn.Parameter(torch.from_numpy(
            np.linspace(vmin, vmax, n).astype('float32')).to(device).view(1, n),
            requires_grad = False)
        self.p = nn.Parameter(torch.randn([n + 1]).to(device) / 5)
        
    def to_positive(self, x):
        return torch.exp(x) + 1e-3

    def forward(self, eps):
        # bias term
        delta_h = self.int_length * self.to_positive(self.p[1:self.n]).detach()
        delta_bias = torch.zeros([self.n]).to(eps.device)
        delta_bias[0] = self.b
        for i in range(self.n-1):
            delta_bias[i+1] = delta_bias[i] + delta_h[i]
            
        index = torch.sum(((eps - self.points) >= 0).long(), 1).detach() 
        start_points = index - 1 # where indicator becomes 1 (index)
        start_points[start_points < 0] = 0 # smaller than boundary, set to zero 
        delta_bias = delta_bias[start_points]
        
        start_points = torch.squeeze(self.points)[torch.squeeze(start_points)].detach() # points where indicator becomes 1
        
        # weight term
        w = self.to_positive(self.p[index])
        
        out = (eps - start_points.view(-1,1)) * w.view(-1,1) + delta_bias.view(-1,1)
        return out

    def inverse(self, out):
        # bias term
        delta_h = self.int_length * self.to_positive(self.p[1:self.n]).detach()
        delta_bias = torch.zeros([self.n]).to(out.device)
        delta_bias[0] = self.b
        for i in range(self.n-1):
            delta_bias[i+1] = delta_bias[i] + delta_h[i]
            
        index = torch.sum(((out - delta_bias) >= 0).long(), 1).detach() 
        start_points = index - 1
        start_points[start_points < 0] = 0
        delta_bias = delta_bias[start_points]
        
        start_points = torch.squeeze(self.points)[torch.squeeze(start_points)].detach()
        
        # weight term
        w = self.to_positive(self.p[index])
        
        eps = (out - delta_bias.view(-1,1)) / w.view(-1,1) + start_points.view(-1,1)
        return eps
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
        if config["node_dim"] == 1:
            if config["scm"] == "linear":
                self.transform = [InvertiblePriorLinear(device=device) for _ in range(config["node"])]
            else:
                self.transform = [InvertiblePWL(device=device) for _ in range(config["node"])]
        else:
            self.transform = {}
            for i in range(config["node"]):
                if config["scm"] == "linear":
                    self.transform[i] = [InvertiblePriorLinear(device=device) for _ in range(config["node"])]
                else:
                    self.transform[i] = [InvertiblePWL(device=device) for _ in range(config["node"])]
        
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
        if self.config["node_dim"] == 1:
            inverse_latent = list(map(lambda x, layer: layer.inverse(x), input, self.transform)) # [batch, 1] x node
        else:
            inverse_latent = []
            for i, z in enumerate(input):
                z_ = torch.split(z, 1, dim=1)
                inverse_latent.append(torch.cat(list(map(lambda x, layer: layer.inverse(x), z_, self.transform[i])), dim=1)) # [batch, node_dim] x node
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
        latent_ = [x.squeeze(dim=2) for x in torch.split(latent, 1, dim=2)] # [batch, node_dim] x node
        
        if self.config["node_dim"] == 1:
            latent = list(map(lambda x, layer: layer(x), latent_, self.transform))# [batch, 1] x node
        else:
            latent = []
            for i, z in enumerate(latent_):
                z_ = torch.split(z, 1, dim=1)
                latent.append(torch.cat(list(map(lambda x, layer: layer(x), z_, self.transform[i])), dim=1)) # [batch, node_dim] x node
        
        xhat = self.decoder(torch.cat(latent, dim=1))
        xhat = xhat.view(-1, self.config["image_size"], self.config["image_size"], 3)
        
        """Alignment"""
        mean_ = mean.view(-1, self.config["node_dim"], self.config["node"]).contiguous() # deterministic part
        align_latent = torch.matmul(mean_, I_B_inv) # [batch, node_dim, node]
        align_latent_ = [x.squeeze(dim=2) for x in torch.split(align_latent, 1, dim=2)] # [batch, node_dim] x node
        
        if self.config["node_dim"] == 1:
            align_latent = list(map(lambda x, layer: layer(x), align_latent_, self.transform)) # [batch, 1] x node
        else:
            align_latent = []
            for i, z in enumerate(align_latent_):
                z_ = torch.split(z, 1, dim=1)
                align_latent.append(torch.cat(list(map(lambda x, layer: layer(x), z_, self.transform[i])), dim=1)) # [batch, node_dim] x node
        
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
        "scm": 'linear'
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
# I = torch.eye(config["node"]).to(device)
# I_B_inv = torch.inverse(I - B)

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
# mean, logvar = torch.split(h, config["node_dim"] * config["node"], dim=1)

# """Latent Generating Process"""
# noise = torch.randn(image.size(0), config["node_dim"] * config["node"]).to(device) 
# epsilon = mean + torch.exp(logvar / 2) * noise
# epsilon = epsilon.view(-1, config["node_dim"], config["node"]).contiguous()
# latent = torch.matmul(epsilon, I_B_inv) # [batch, node_dim, node]
# latent_orig = latent.clone()
# latent = [x.squeeze(dim=2) for x in torch.split(latent, 1, dim=2)] # [batch, node_dim] x node

# latent = list(map(lambda x, layer: layer(x), latent, flows))

# xhat = decoder(torch.cat(latent, dim=1))
# xhat = xhat.view(-1, 96, 96, 3)

# """Alignment"""
# mean_ = mean.view(-1, config["node_dim"], config["node"]).contiguous()
# align_latent = torch.matmul(mean_, I_B_inv) # [batch, node_dim, node]
# align_latent = [x.squeeze(dim=2) for x in torch.split(align_latent, 1, dim=2)] # [batch, node_dim] x node
# align_latent = list(map(lambda x, layer: layer(x), align_latent, flows))
#%%