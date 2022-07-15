#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

from torch.autograd import Variable
#%%
class GAE(nn.Module):
    def __init__(self, config):
        super(GAE, self).__init__()
        
        self.config = config
        
        # encoder 
        encoder = []
        in_dim = config["x_dim"]
        for j in range(config["num_layer"]):
            encoder.append(nn.Linear(in_dim, config["hidden_dim"]))
            encoder.append(nn.LeakyReLU(0.05))
            in_dim = config["hidden_dim"]
        encoder.append(nn.Linear(in_dim, config["latent_dim"]))
        self.encoder = nn.ModuleList(encoder)
        
        # decoder
        decoder = []
        in_dim = config["latent_dim"]
        for j in range(config["num_layer"]):
            decoder.append(nn.Linear(in_dim, config["hidden_dim"]))
            decoder.append(nn.LeakyReLU(0.05))
            in_dim = config["hidden_dim"]
        decoder.append(nn.Linear(in_dim, config["x_dim"]))
        self.decoder = nn.ModuleList(decoder)
        
        # self.init_weights()
        
        # weighted adjacency matrix
        W = torch.rand(config["d"], config["d"])
        min = -0.1
        max = 0.1
        W = (max - min) * W + min # ~ Uniform(-0.1, 0.1)
        W = W.fill_diagonal_(0.)
        self.W = nn.Parameter(W, requires_grad=True)
    
    # def init_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Linear):
    #             nn.init.xavier_normal_(m.weight.data)
    #         elif isinstance(m, nn.BatchNorm1d):
    #             m.weight.data.fill_(1)
    #             m.bias.data.zero_()
    
    def forward(self, input):
        h = input.reshape(-1, self.config["x_dim"])
        # encoder
        for e in self.encoder:
            h = e(h)
        # causal structure
        h = h.reshape(-1, self.config["d"], self.config["latent_dim"])
        h = torch.matmul(self.W.t(), h)
        h = h.reshape(-1, self.config["latent_dim"])
        # decoder
        for d in self.decoder:
            h = d(h)
        h = h.reshape(-1, self.config["d"], self.config["x_dim"])
        return h
#%%
def main():
    config = {
        "n": 100,
        "d": 7,
        "x_dim": 5,
        "latent_dim": 3,
        "num_layer": 2,
        "hidden_dim": 16,
    }
    
    model = GAE(config)
    for x in model.parameters():
        print(x)
    assert torch.trace(model.W) == 0.
    
    batch = torch.rand(config["n"], config["d"], config["x_dim"])
    recon = model(batch)
    assert recon.shape == (config["n"], config["d"], config["x_dim"])
    
    print("Model test pass!")
#%%
if __name__ == '__main__':
    main()
#%%