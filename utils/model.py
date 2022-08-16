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
                 config,
                 device='cpu'):
        super(PlanarFlows, self).__init__()

        self.config = config
        
        self.w = [(nn.Parameter(torch.randn(config["node_dim"], 1) * 0.1).to(device))
                  for _ in range(config["flow_num"])]
        self.b = [nn.Parameter((torch.randn(1, 1) * 0.1).to(device))
                  for _ in range(config["flow_num"])]
        self.u = [nn.Parameter((torch.randn(config["node_dim"], 1) * 0.1).to(device))
                  for _ in range(config["flow_num"])]
        
    def build_u(self, u_, w_):
        """sufficient condition to be invertible"""
        term1 = -1 + torch.log(1 + torch.exp(w_.t() @ u_))
        term2 = w_.t() @ u_
        u_hat = u_ + (term1 - term2) * (w_ / torch.norm(w_, p=2) ** 2)
        return u_hat
    
    def inverse(self, inputs):
        h = inputs
        for j in reversed(range(self.config["flow_num"])):
            z = h
            for _ in range(self.config["inverse_loop"]):
                u_ = self.build_u(self.u[j], self.w[j]) 
                z = h - u_.t() * torch.tanh(z @ self.w[j] + self.b[j])
            h = z
        return h
            
    def forward(self, inputs):
        h = inputs
        for j in range(self.config["flow_num"]):
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
            nn.Linear(300, config["node"] * config["node_dim"]),
            nn.BatchNorm1d(config["node"] * config["node_dim"])
        ).to(device)
        
        self.B = B.to(device) # binary adjacency matrix
        # self.batchnorm = nn.BatchNorm1d(config["node"] * config["embedding_dim"]).to(device)
        
        """Generalized Linear SEM: Invertible NN"""
        self.flows = [PlanarFlows(config, device)
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
        self.prior_nn = [nn.Sequential(
                        nn.Linear(1, 4),
                        nn.Tanh(),
                        nn.Linear(4, config["node_dim"]),).to(device) 
                        for _ in range(config["node"])]
        
        # """Alignment"""
        
        self.I = torch.eye(config["node"]).to(device)

    def inverse(self, input):
        inverse_latent = list(map(lambda x, layer: layer.inverse(torch.atanh(x)), input, self.flows))
        return inverse_latent
    
    def forward(self, input):
        image, label = input
        logvar = self.encoder(nn.Flatten()(image)) # [batch, node * node_dim]
        
        """Latent Generating Process"""
        epsilon = torch.exp(logvar / 2) * torch.randn(image.size(0), self.config["node"] * self.config["node_dim"]).to(self.device) 
        epsilon = epsilon.view(-1, self.config["node_dim"], self.config["node"]).contiguous()
        latent = torch.matmul(epsilon, torch.inverse(self.I - self.B)) # [batch, node_dim, node]
        latent_orig = latent.clone()
        latent = [x.squeeze(dim=2) for x in torch.split(latent, 1, dim=2)] # [batch, node_dim] x node1
        causal_latent = list(map(lambda x, layer: torch.tanh(layer(x)), latent, self.flows))

        xhat = self.decoder(torch.cat(causal_latent, dim=1))
        xhat = xhat.view(-1, 96, 96, 3)

        """prior"""
        prior_logvar = list(map(lambda x, layer: layer(x), torch.split(label, 1, dim=1), self.prior_nn))
        prior_logvar = torch.cat(prior_logvar, dim=1)
        
        return logvar, prior_logvar, latent_orig, causal_latent, xhat
#%%
def main():
    config = {
        "n": 10,
        "node": 4,
        "node_dim": 1, 
        "flow_num": 3,
        "inverse_loop": 100,
    }
    
    B = torch.zeros(config["node"], config["node"])
    B[:2, 2:] = 1
    model = VAE(B, config, 'cpu')
    for x in model.parameters():
        print(x.shape)
        
    batch = torch.rand(config["n"], 96, 96, 3)
    u = torch.rand(config["n"], config["node"])
    logvar, prior_logvar, latent_orig, causal_latent, xhat = model([batch, u])
    
    assert logvar.shape == (config["n"], config["node"] * config["node_dim"])
    assert prior_logvar.shape == (config["n"], config["node"] * config["node_dim"])
    assert latent_orig.shape == (config["n"], config["node_dim"], config["node"])
    assert causal_latent[0].shape == (config["n"], config["node_dim"])
    assert len(causal_latent) == config["node"]
    assert xhat.shape == (config["n"], 96, 96, 3)
    
    inverse_diff = torch.abs(sum([x - y for x, y in zip([x.squeeze(dim=2) for x in torch.split(latent_orig, 1, dim=2)], 
                                                        model.inverse(causal_latent))]).sum())
    assert inverse_diff / (config["n"] * config["node"] * config["node_dim"])  < 1e-5
    
    print("Model test pass!")
#%%
if __name__ == '__main__':
    main()
#%%
# config = {
#     "n": 10,
#     "node": 4,
#     "node_dim": 2,
#     "align_dim": 4, 
#     "flow_num": 3,
#     "inverse_loop": 100,
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
#     nn.Linear(300, config["node"] * config["node_dim"]),
# ).to(device)

# B = B.to(device) # weighted adjacency matrix
# # batchnorm = nn.BatchNorm1d(config["node"] * config["embedding_dim"]).to(device)

# flows = [PlanarFlows(config, device)
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

# I = torch.eye(config["node"]).to(device)

# """alignment net"""
# alignnet = [nn.Sequential(
#     nn.Linear(1, config["align_dim"]),
#     nn.Tanh(),
#     nn.Linear(config["align_dim"], 1),).to(device) 
#     for _ in range(config["node"])]
# #%%
# logvar = encoder(nn.Flatten()(x)) # [batch, node*embedding_dim]

# """Latent Generating Process"""
# epsilon = torch.exp(logvar / 2) * torch.randn(x.size(0), config["node"] * config["node_dim"])
# epsilon = epsilon.view(-1, config["node_dim"], config["node"]).contiguous()

# latent = torch.matmul(epsilon, torch.inverse(I - B))
# latent_orig = latent.clone()
# latent = [x.squeeze(dim=2) for x in torch.split(latent, 1, dim=2)]

# causal_latent = list(map(lambda x, layer: torch.tanh(layer(x)), latent, flows))

# xhat = decoder(torch.cat(causal_latent, dim=1))
# xhat = xhat.view(-1, 96, 96, 3)

# """prior"""
# prior_logvar = list(map(lambda x, layer: layer(x), torch.split(u, 1, dim=1), alignnet))
# prior_logvar = torch.cat(prior_logvar, dim=1)

# # inverse_latent = list(map(lambda x, layer: layer.inverse(torch.atanh(x)), causal_latent, flows))
# # [x - y for x, y in zip(inverse_latent, latent)]
# #%%
# # causal_latent = []
# # for j in range(config["node"]):
# #     u_ = build_u(u[j], w[j])
# #     causal_latent.append(latent[j] + u_.t() * torch.tanh(latent[j] @ w[j] + b[j]))
#%%