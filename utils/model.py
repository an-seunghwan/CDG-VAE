#%%
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
#%%
class VectorQuantizer(nn.Module):
    """
    Reference:
    [1] https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
    [2] https://github.com/AntixK/PyTorch-VAE/blob/a6896b944c918dd7030e7d795a8c13e5c6345ec7/models/vq_vae.py#L2
    """
    def __init__(self, config, device):
        super(VectorQuantizer, self).__init__()
        
        self.config = config
        self.device = device
        
        self.embedding = nn.Embedding(config["num_embeddings"], config["embedding_dim"]).to(device)
        self.embedding.weight.data.uniform_(-1/config["num_embeddings"], 1/config["num_embeddings"]).to(device)
        
    def forward(self, latents):
        latents = latents.to(self.device)
        latents = latents.view(-1, self.config["node"], self.config["embedding_dim"]).contiguous()
        flat_latents = latents.view(-1, self.config["embedding_dim"]) # [batch*node x embedding_dim]

        # Compute L2 distance between latents and embedding weights
        dist = torch.sum(flat_latents ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1) - \
            2 * torch.matmul(flat_latents, self.embedding.weight.t()) # [batch*node x embedding_dim]

        # Get the encoding that has the min distance
        encoding_index = torch.argmin(dist, dim=1).unsqueeze(1)

        # Convert to one-hot encodings
        encoding_one_hot = torch.zeros(encoding_index.size(0), self.config["num_embeddings"]).to(self.device)
        # dim: one hot encoding dimension
        # index: position of src
        # value: the value of corresponding index
        encoding_one_hot.scatter_(dim=1, index=encoding_index, value=1)

        # Quantize the latents
        quantized_latent = torch.matmul(encoding_one_hot, self.embedding.weight) # [batch*node, D]
        quantized_latent = quantized_latent.view(latents.shape)

        # Compute the VQ losses
        embedding_loss = F.mse_loss(latents.detach(), quantized_latent) # training embeddings
        commitment_loss = F.mse_loss(latents, quantized_latent.detach()) # prevent encoder from growing
        vq_loss = embedding_loss + self.config["beta"] * commitment_loss

        # Add the residue back to the latents (straight-through gradient estimation)
        quantized_latent = latents + (quantized_latent - latents).detach()
        
        return quantized_latent, vq_loss 
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
            nn.Linear(300, config["node"] * config["embedding_dim"]),
            nn.BatchNorm1d(config["node"] * config["embedding_dim"]),
        ).to(device)
        
        """Build Vector Quantizer"""
        self.vq_layer = VectorQuantizer(self.config, device).to(device)
        
        self.B = B # weighted adjacency matrix
        # self.batchnorm = nn.BatchNorm1d(config["node"] * config["embedding_dim"]).to(device)
        
        """decoder"""
        self.decoder = nn.Sequential(
            nn.Linear(config["node"] * config["embedding_dim"], 300),
            nn.ELU(),
            nn.Linear(300, 300),
            nn.ELU(),
            nn.Linear(300, 3*96*96),
            nn.Tanh()
        ).to(device)

        self.I = torch.eye(config["node"]).to(device)
        
        # u = 0.5 * torch.log((1. + u) / (1. - u) + 1e-8) # tanh inverse function
        
    def forward(self, input):
        latent = self.encoder(nn.Flatten()(input)) # [batch, node*embedding_dim]
        
        """Vector Quantization w.r.t. exogenous variables"""
        quantized_latent, vq_loss = self.vq_layer(latent)
        quantized_latent = quantized_latent.permute(0, 2, 1).contiguous() # [batch, embedding_dim, node]

        """Latent Generating Process"""
        causal_latent = torch.matmul(quantized_latent, torch.inverse(self.I - self.B))
        # causal_latent = self.batchnorm(causal_latent)
        causal_latent_orig = causal_latent.clone() # save for DAG reconstruction loss
        causal_latent = torch.tanh(causal_latent) # intervention range (-1, 1)
        causal_latent = causal_latent.permute(0, 2, 1).contiguous() # [batch, node, embedding_dim]
        causal_latent = causal_latent.view(-1, self.config["node"] * self.config["embedding_dim"]).contiguous()

        xhat = self.decoder(causal_latent).view(-1, 96, 96, 3)
        return causal_latent_orig, causal_latent, xhat, vq_loss
#%%
def main():
    config = {
        "n": 10,
        "num_embeddings": 10,
        "node": 4,
        "embedding_dim": 1,
        "beta": 0.25,
    }
    
    B = torch.randn(config["node"], config["node"]) / 10 + torch.eye(config["node"])
    model = VAE(B, config, 'cpu')
    for x in model.parameters():
        print(x.shape)
        
    batch = torch.rand(config["n"], 96, 96, 3)
    causal_latent_orig, causal_latent, xhat, vq_loss = model(batch)
    
    assert causal_latent_orig.shape == (config["n"], config["embedding_dim"], config["node"])
    assert causal_latent.shape == (config["n"], config["node"] * config["embedding_dim"])
    assert xhat.shape == (config["n"], 96, 96, 3)
    
    inversed_latent = 0.5 * torch.log((1. + causal_latent) / (1. - causal_latent) + 1e-8) # tanh inverse function
    
    assert torch.isclose(causal_latent_orig.squeeze(dim=1), inversed_latent).sum()
    # assert torch.abs(causal_latent_orig.squeeze(dim=1) - inversed_latent).sum() < 1e-4
    
    print("Model test pass!")
#%%
if __name__ == '__main__':
    main()
#%%
# config = {
#     "n": 10,
#     "num_embeddings": 10,
#     "node": 4,
#     "embedding_dim": 1,
#     "beta": 0.25,
# }

# B = torch.zeros(config["node"], config["node"])
# B[:2, 2:] = 1

# x = torch.randn(config["n"], 96, 96, 3)

# """encoder"""
# encoder = nn.Sequential(
#     nn.Linear(3*96*96, 300),
#     nn.ELU(),
#     nn.Linear(300, 300),
#     nn.ELU(),
#     nn.Linear(300, config["node"] * config["embedding_dim"]),
# )

# latents = encoder(nn.Flatten()(x))
# #%%
# ##### VQ
# latents = latents.view(-1, config["node"], config["embedding_dim"]).contiguous()

# embedding = nn.Embedding(config["num_embeddings"], config["embedding_dim"])
# embedding.weight.data.uniform_(-1/config["num_embeddings"], 1/config["num_embeddings"])

# flat_latents = latents.view(-1, config["embedding_dim"]) # [batch*node x embedding_dim]

# # Compute L2 distance between latents and embedding weights
# dist = torch.sum(flat_latents ** 2, dim=1, keepdim=True) + \
#     torch.sum(embedding.weight ** 2, dim=1) - \
#     2 * torch.matmul(flat_latents, embedding.weight.t()) # [batch*node x embedding_dim]

# # Get the encoding that has the min distance
# encoding_index = torch.argmin(dist, dim=1).unsqueeze(1)

# # Convert to one-hot encodings
# encoding_one_hot = torch.zeros(encoding_index.size(0), config["num_embeddings"])
# # dim: one hot encoding dimension
# # index: position of src
# # value: the value of corresponding index
# encoding_one_hot.scatter_(dim=1, index=encoding_index, value=1)

# # Quantize the latents
# quantized_latent = torch.matmul(encoding_one_hot, embedding.weight) # [batch*node x embedding_dim]
# quantized_latent = quantized_latent.view(latents.shape)

# # Compute the VQ losses
# embedding_loss = F.mse_loss(latents.detach(), quantized_latent) # training embeddings
# commitment_loss = F.mse_loss(latents, quantized_latent.detach()) # prevent encoder from growing
# vq_loss = embedding_loss + config["beta"] * commitment_loss

# # Add the residue back to the latents (straight-through gradient estimation)
# quantized_latent = latents + (quantized_latent - latents).detach()
# #%%
# quantized_latent = quantized_latent.permute(0, 2, 1).contiguous()

# """Latent Generating Process"""
# I = torch.eye(config["node"])
# causal_latent = torch.matmul(quantized_latent, torch.inverse(I - B))
# causal_latent = causal_latent.permute(0, 2, 1).contiguous()

# causal_latent = causal_latent.view(-1, config["node"] * config["embedding_dim"])
#%%