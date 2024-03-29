#%%
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
#%%
class InvertiblePriorLinear(nn.Module):
    """Invertible Prior for Linear case
    Reference:
    [1]: https://github.com/xwshen51/DEAR/blob/main/causal_model.py

    Parameter:
        p: mean and std parameter for scaling
    """
    def __init__(self, device='cpu'):
        super(InvertiblePriorLinear, self).__init__()
        self.p = nn.Parameter(torch.rand([2], device=device) * 0.1)

    def forward(self, eps, log_determinant=False):
        o = self.p[0] * eps + self.p[1]
        logdet = 0
        if log_determinant:
            logdet += torch.log(self.p[0].abs()).repeat(eps.size(0), 1)
        return o, logdet
    
    def inverse(self, o):
        eps = (o - self.p[1]) / self.p[0]
        return eps
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
        self.device = device
        self.alpha = torch.tensor(1, dtype=torch.float32).to(device) # parameter of ELU
        
        self.w = nn.ParameterList(
            [(nn.Parameter(torch.randn(self.input_dim, 1, device=device) * 0.1))
            for _ in range(self.flow_num)])
        self.b = nn.ParameterList(
            [nn.Parameter((torch.randn(1, 1, device=device) * 0.1))
            for _ in range(self.flow_num)])
        self.u = nn.ParameterList(
            [nn.Parameter((torch.randn(self.input_dim, 1, device=device) * 0.1))
            for _ in range(self.flow_num)])
        
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
                gradient = torch.where(x > torch.tensor(0, dtype=torch.float32).to(self.device), 
                                       torch.tensor(1, dtype=torch.float32).to(self.device), 
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
            nn.Linear(300, config["node"] * 2),
        ).to(device)
        
        """Causal Adjacency Matrix"""
        self.B = B.to(device) 
        self.I = torch.eye(config["node"]).to(device)
        self.I_B_inv = torch.inverse(self.I - self.B)
        
        """Generalized Linear SEM: Invertible NN"""
        if config["scm"] == "linear":
            self.flows = nn.ModuleList(
                [InvertiblePriorLinear(device=device) for _ in range(config["node"])])
        elif config["scm"] == "nonlinear":
            self.flows = nn.ModuleList(
                [PlanarFlows(1, config["flow_num"], config["inverse_loop"], device) for _ in range(config["node"])])
        else:
            raise ValueError('Not supported SCM!')
        
        """decoder"""
        self.decoder = nn.Sequential(
            nn.Linear(config["node"], 300),
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
        h = self.encoder(nn.Flatten()(input)) # [batch, node * 2]
        mean, logvar = torch.split(h, self.config["node"], dim=1)
        return mean, logvar
    
    def transform(self, input, log_determinant=False):
        latent = torch.matmul(input, self.I_B_inv) # [batch, node], input = epsilon (exogenous variables)
        orig_latent = latent.clone()
        latent = torch.split(latent, 1, dim=1) # [batch, 1] x node
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
            noise = torch.randn(input.size(0), self.config["node"]).to(self.device) 
            epsilon = mean + torch.exp(logvar / 2) * noise
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
class Discriminator(nn.Module):
    def __init__(self, config, device='cpu'):
        super(Discriminator, self).__init__()
        self.config = config
        self.net = nn.Sequential(
            nn.Linear(3*config["image_size"]*config["image_size"] + config["node"], 300),
            nn.ELU(),
            nn.Linear(300, 300),
            nn.ELU(),
            nn.Linear(300, 1),
        ).to(device)

    def forward(self, x, z):
        x = x.view(-1, 3*self.config["image_size"]*self.config["image_size"])
        x = torch.cat((x, z), dim=1)
        return self.net(x)
#%%
class CDGVAE(nn.Module):
    def __init__(self, B, mask, config, device):
        super(CDGVAE, self).__init__()
        
        self.config = config
        self.mask = mask
        # assert sum(config["factor"]) == config["node"]
        assert len(config["factor"]) == len(mask)
        self.device = device
        
        """encoder"""
        self.encoder = nn.Sequential(
            nn.Linear(3*config["image_size"]*config["image_size"], 300),
            nn.ELU(),
            nn.Linear(300, 300),
            nn.ELU(),
            nn.Linear(300, config["node"] * 2),
        ).to(device)
        
        """Causal Adjacency Matrix"""
        self.B = B.to(device) 
        self.I = torch.eye(config["node"]).to(device)
        self.I_B_inv = torch.inverse(self.I - self.B)
        
        """Generalized Linear SEM: Invertible NN"""
        if config["scm"] == "linear":
            self.flows = nn.ModuleList(
                [InvertiblePriorLinear(device=device) for _ in range(config["node"])])
        elif config["scm"] == "nonlinear":
            self.flows = nn.ModuleList(
                [PlanarFlows(1, config["flow_num"], config["inverse_loop"], device) for _ in range(config["node"])])
        else:
            raise ValueError('Not supported SCM!')
        
        """decoder"""
        self.decoder = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(k+1, 300),
                nn.ELU(),
                nn.Linear(300, 300),
                nn.ELU(),
                nn.Linear(300, 3*config["image_size"]*config["image_size"]),
            ).to(device) for k in config["factor"]])
        
    def inverse(self, input): 
        inverse_latent = list(map(lambda x, layer: layer.inverse(x), input, self.flows))
        return inverse_latent
    
    def get_posterior(self, input):
        h = self.encoder(nn.Flatten()(input)) # [batch, node * 2]
        mean, logvar = torch.split(h, self.config["node"], dim=1)
        return mean, logvar
    
    def transform(self, input, log_determinant=False):
        latent = torch.matmul(input, self.I_B_inv) # [batch, node], input = epsilon (exogenous variables)
        orig_latent = latent.clone()
        latent = torch.split(latent, 1, dim=1) # [batch, 1] x node
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
            noise = torch.randn(input.size(0), self.config["node"]).to(self.device) 
            epsilon = mean + torch.exp(logvar / 2) * noise
        orig_latent, latent, logdet = self.transform(epsilon, log_determinant=log_determinant)
        return mean, logvar, epsilon, orig_latent, latent, logdet
    
    def decode(self, input):
        latent = torch.cat(input, axis=1)
        
        """DR version"""
        spurious = latent[:, [-1]]
        latent = torch.split(latent[:, :-1], self.config["factor"], dim=-1)
        latent = [torch.cat([z, spurious], dim=1) for z in latent]
        
        xhat_separated = [D(z) for D, z in zip(self.decoder, latent)]
        xhat = [x.view(-1, self.config["image_size"], self.config["image_size"], 3) for x in xhat_separated]
        xhat = [x * m.to(self.device) for x, m in zip(xhat, self.mask)] # masking
        xhat = torch.tanh(sum(xhat)) 
        return xhat_separated, xhat
    
    def forward(self, input, deterministic=False, log_determinant=False):
        """encoding"""
        mean, logvar, epsilon, orig_latent, latent, logdet = self.encode(input, 
                                                                         deterministic=deterministic,
                                                                         log_determinant=log_determinant)
        
        """decoding"""
        xhat_separated, xhat = self.decode(latent)
        
        """Alignment"""
        _, _, _, _, align_latent, _ = self.encode(input, 
                                                  deterministic=True, 
                                                  log_determinant=log_determinant)
        
        return mean, logvar, epsilon, orig_latent, latent, logdet, align_latent, xhat_separated, xhat
#%%
class DownstreamClassifier(nn.Module):
    def __init__(self, config, device):
        super(DownstreamClassifier, self).__init__()
        
        self.config = config
        self.device = device
        
        self.classify = nn.Sequential(
            nn.Linear(config["node"] - 1, 2),
            nn.ELU(),
            nn.Linear(2, 1),
            nn.Sigmoid()
        ).to(device)
        
    def forward(self, input):
        out = self.classify(input)
        return out
#%%