#%%
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

# from module.resnet import *
from torchvision import models
from module.sagan import Generator
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
class CDGVAE(nn.Module):
    def __init__(self, B, mask, config, device, fc_size=32):
        super(CDGVAE, self).__init__()
        
        self.config = config
        self.mask = mask
        # assert sum(config["factor"]) == config["node"]
        # assert len(config["factor"]) == len(mask)
        self.device = device
        
        """encoder"""
        self.encoder = models.resnet18(pretrained=True)
        self.encoder.fc = nn.Linear(
            self.encoder.fc.in_features, 
            config["node"] * 2 + config["latent_dim"] * 2)
        """freeze!"""
        for param in self.encoder.parameters():
            param.requires_grad_(False)
        self.encoder.fc.weight.requires_grad = True
        self.encoder.fc.bias.requires_grad = True
        # self.encoder = resnet18(
        #     pretrained=True, in_channels=3, fc_size=fc_size, 
        #     out_dim=config["node"] * 2 + config["latent_dim"] * 2).to(device)
        
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
        self.decoder = [
            Generator(2).to(device), 
            Generator(2).to(device), 
            Generator(2).to(device), 
            Generator(3).to(device), 
            Generator(config["latent_dim"]).to(device)]
        
    def inverse(self, input): 
        inverse_latent = list(map(lambda x, layer: layer.inverse(x), input, self.flows))
        return inverse_latent
    
    def get_posterior(self, input):
        h = self.encoder(input[..., :3].permute(0, 3, 1, 2))
        h1, h2 = torch.split(h, [self.config["node"] * 2, self.config["latent_dim"] * 2], dim=1)
        mean1, logvar1 = torch.split(h1, self.config["node"], dim=1)
        mean2, logvar2 = torch.split(h2, self.config["latent_dim"], dim=1)
        # h = self.encoder(nn.Flatten()(input)) # [batch, node * 2]
        # mean, logvar = torch.split(h, self.config["node"], dim=1)
        return mean1, logvar1, mean2, logvar2
    
    def transform(self, input, log_determinant=False):
        latent = torch.matmul(input, self.I_B_inv) # [batch, node], input = epsilon (exogenous variables)
        orig_latent = latent.clone()
        latent = torch.split(latent, 1, dim=1) # [batch, 1] x node
        latent = list(map(lambda x, layer: layer(x, log_determinant=log_determinant), latent, self.flows)) # input = (I-B^T)^{-1} * epsilon
        logdet = [x[1] for x in latent]
        latent = [x[0] for x in latent]
        return orig_latent, latent, logdet
    
    def encode(self, input, deterministic=False, log_determinant=False):
        mean1, logvar1, mean2, logvar2 = self.get_posterior(input)
        """Latent Generating Process"""
        if deterministic:
            epsilon1 = mean1
            epsilon2 = mean2
        else:
            noise = torch.randn(input.size(0), self.config["node"]).to(self.device) 
            epsilon1 = mean1 + torch.exp(logvar1 / 2) * noise
            noise = torch.randn(input.size(0), self.config["node"]).to(self.device) 
            epsilon2 = mean2 + torch.exp(logvar2 / 2) * noise
        orig_latent, latent, logdet = self.transform(epsilon1, log_determinant=log_determinant)
        return (mean1, logvar1, epsilon1, orig_latent, latent, logdet), (mean2, logvar2, epsilon2)
    
    def decode(self, latent, epsilon2):
        latent = [
            torch.cat([latent[0], latent[2]], dim=1),
            torch.cat([latent[0], latent[3]], dim=1),
            torch.cat([latent[0], latent[4]], dim=1),
            torch.cat([latent[0], latent[1], latent[5]], dim=1),
            epsilon2]
        xhat_separated = [D(z) for D, z in zip(self.decoder, latent)]
        xhat = [x.permute(0, 2, 3, 1) for x in xhat_separated]
        xhat = [x * m.to(self.device) for x, m in zip(xhat, self.mask)] # masking
        xhat = torch.tanh(sum(xhat)) # generalized addictive model (GAM)
        return xhat_separated, xhat
    
    def forward(self, input, deterministic=False, log_determinant=False):
        """encoding"""
        (mean1, logvar1, epsilon1, orig_latent, latent, logdet), (mean2, logvar2, epsilon2) = self.encode(
            input, 
            deterministic=deterministic,
            log_determinant=log_determinant)
        
        """decoding"""
        xhat_separated, xhat = self.decode(latent, epsilon2)
        
        """Alignment"""
        (_, _, _, _, align_latent, _), _ = self.encode(
            input, 
            deterministic=True, 
            log_determinant=log_determinant)
        
        return (mean1, logvar1, epsilon1, orig_latent, latent, logdet), (mean2, logvar2, epsilon2), align_latent, xhat_separated, xhat
#%%