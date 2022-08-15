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
        flat_latents = latents.view(-1, self.config["embedding_dim"]) # [batch x embedding_dim]

        # Compute L2 distance between latents and embedding weights
        dist = torch.sum(flat_latents ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1) - \
            2 * torch.matmul(flat_latents, self.embedding.weight.t()) # [batch x embedding_dim]

        # Get the encoding that has the min distance
        encoding_index = torch.argmin(dist, dim=1).unsqueeze(1)

        # Convert to one-hot encodings
        encoding_one_hot = torch.zeros(encoding_index.size(0), self.config["num_embeddings"]).to(self.device)
        # dim: one hot encoding dimension
        # index: position of src
        # value: the value of corresponding index
        encoding_one_hot.scatter_(dim=1, index=encoding_index, value=1)

        # Quantize the latents
        quantized_latent = torch.matmul(encoding_one_hot, self.embedding.weight) # [batch x embedding_dim]
        quantized_latent = quantized_latent.view(latents.shape)

        # Compute the VQ losses
        embedding_loss = F.mse_loss(latents.detach(), quantized_latent) # training embeddings
        commitment_loss = F.mse_loss(latents, quantized_latent.detach()) # prevent encoder from growing
        vq_loss = embedding_loss + self.config["beta"] * commitment_loss

        # Add the residue back to the latents (straight-through gradient estimation)
        quantized_latent = latents + (quantized_latent - latents).detach()
        
        return quantized_latent, vq_loss 
#%%
# class AlignNet(nn.Module):
#     def __init__(self, config, device, output_dim=1, hidden_dim=4):
#         super(AlignNet, self).__init__()
        
#         self.config = config
#         self.device = device
        
#         self.net = [
#             nn.Sequential(
#                 nn.Linear(config["embedding_dim"], hidden_dim),
#                 nn.ReLU(),
#                 nn.Linear(hidden_dim, output_dim)
#                 ).to(device) 
#             for _ in range(config["node"])]
        
#     def forward(self, input):
#         split_input = list(map(lambda x: x.squeeze(dim=1), torch.split(input, 1, dim=1)))
#         h = list(map(lambda x, layer: layer(x), split_input, self.net))
#         h = torch.cat(h, dim=1)
#         return h
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
        self.vq_layer = [VectorQuantizer(config, device).to(device)
                       for _ in range(config["node"])]
        
        self.B = B.to(device) # binary adjacency matrix
        # self.batchnorm = nn.BatchNorm1d(config["node"] * config["embedding_dim"]).to(device)
        
        """Shared layer"""
        self.shared = [nn.Sequential(
                nn.Linear(config["node"] * config["embedding_dim"] + config["embedding_dim"], config["hidden_dim"]),
                nn.ReLU(),
            ) for _ in range(config["node"])]
        
        """NPSEM: NO assumptions"""
        self.npsem = [nn.Sequential(
                nn.Linear(config["hidden_dim"], config["embedding_dim"]),
                nn.Tanh(),
            ) for _ in range(config["node"])]
        
        """Alignment"""
        self.alignnet = [nn.Linear(config["hidden_dim"], 1).to(device) for _ in range(config["node"])]
        
        """decoder"""
        self.decoder = nn.Sequential(
            nn.Linear(config["node"] * config["embedding_dim"], 300),
            nn.ELU(),
            nn.Linear(300, 300),
            nn.ELU(),
            nn.Linear(300, 3*96*96),
            nn.Tanh()
        ).to(device)

    def forward(self, input):
        latent = self.encoder(nn.Flatten()(input)) # [batch, node * embedding_dim]
        
        """Vector Quantization of exogenous variables"""
        latent = latent.view(-1, self.config["node"], self.config["embedding_dim"]).contiguous() # [batch, node, embedding_dim]
        latent = [x.squeeze(dim=1) for x in torch.split(latent, 1, dim=1)]
        vq_outputs = list(map(lambda x, layer: layer(x), latent, self.vq_layer))
        
        vq_loss = sum([x[1] for x in vq_outputs])
        exogenous = [x[0] for x in vq_outputs] # [batch, embedding_dim] x node

        """Latent Generating Process"""
        causal_latent = torch.zeros(input.size(0), self.config["node"], self.config["embedding_dim"])
        label_hat = torch.zeros(input.size(0), self.config["node"])
        for j in range(self.config["node"]):
            h = (self.B[:, [j]] * causal_latent).view(-1, self.config["node"] * self.config["embedding_dim"]).contiguous()
            h = self.shared[j](torch.cat([h, exogenous[j]], dim=1))
            causal_latent[:, j, :] = self.npsem[j](h)
            label_hat[:, [j]] = self.alignnet[j](h)

        xhat = self.decoder(causal_latent.view(-1, self.config["node"] * self.config["embedding_dim"]).contiguous())
        xhat = xhat.view(-1, 96, 96, 3)
        
        return causal_latent, xhat, vq_loss, label_hat
#%%
def main():
    config = {
        "n": 10,
        "num_embeddings": 10,
        "node": 4,
        "embedding_dim": 2,
        "beta": 0.25,
        "hidden_dim": 2, 
    }
    
    B = torch.randn(config["node"], config["node"]) / 10 + torch.eye(config["node"])
    model = VAE(B, config, 'cpu')
    for x in model.parameters():
        print(x.shape)
        
    batch = torch.rand(config["n"], 96, 96, 3)
    causal_latent, xhat, vq_loss, label_hat = model(batch)
    
    assert causal_latent.shape == (config["n"], config["node"], config["embedding_dim"])
    assert xhat.shape == (config["n"], 96, 96, 3)
    assert label_hat.shape == (config["n"], config["node"])
    
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
#     "npsem_dim": 2, 
#     "align_dim": 4, 
#     "d": 1,
# }

# B = torch.zeros(config["node"], config["node"])
# B[:2, 2:] = 1

# x = torch.randn(config["n"], 96, 96, 3)

# config = config
# device = 'cpu'

# """encoder"""
# encoder = nn.Sequential(
#     nn.Linear(3*96*96, 300),
#     nn.ELU(),
#     nn.Linear(300, 300),
#     nn.ELU(),
#     nn.Linear(300, config["node"] * config["embedding_dim"]),
#     nn.BatchNorm1d(config["node"] * config["embedding_dim"]),
# ).to(device)

# """Build Vector Quantizer"""
# vq_layer = [VectorQuantizer(config, device).to(device)
#             for _ in range(config["node"])]

# B = B.to(device) # weighted adjacency matrix
# # batchnorm = nn.BatchNorm1d(config["node"] * config["embedding_dim"]).to(device)

# """NPSEM"""
# npsem = [nn.Sequential(
#     nn.Linear(config["node"] * config["embedding_dim"] + config["embedding_dim"], config["npsem_dim"]),
#     nn.ReLU(),
#     nn.Linear(config["npsem_dim"], config["embedding_dim"]),
#     nn.Tanh(),
#     ) for _ in range(config["node"])]

# """decoder"""
# decoder = nn.Sequential(
#     nn.Linear(config["node"] * config["embedding_dim"], 300),
#     nn.ELU(),
#     nn.Linear(300, 300),
#     nn.ELU(),
#     nn.Linear(300, 3*96*96),
#     nn.Tanh()
# ).to(device)

# I = torch.eye(config["node"]).to(device)

# """alignment net"""
# alignnet = AlignNet(config, device, hidden_dim=config["align_dim"])
# #%%
# latent = encoder(nn.Flatten()(x)) # [batch, node*embedding_dim]

# """Vector Quantization of exogenous variables"""
# latent = latent.view(-1, config["node"], config["embedding_dim"]).contiguous()
# latent = [x.squeeze(dim=1) for x in torch.split(latent, 1, dim=1)]
# vq_outputs = list(map(lambda x, layer: layer(x), latent, vq_layer))

# vq_loss = sum([x[1] for x in vq_outputs])
# exogenous = [x[0] for x in vq_outputs]

# """Latent Generating Process"""
# causal_latent = torch.zeros(x.size(0), config["node"], config["embedding_dim"])
# for j in range(config["node"]):
#     h = (B[:, [j]] * causal_latent).view(-1, config["node"] * config["embedding_dim"]).contiguous()
#     causal_latent[:, j, :] = npsem[j](torch.cat([h, exogenous[j]], dim=1))

# xhat = decoder(causal_latent.view(-1, config["node"] * config["embedding_dim"]).contiguous())
# xhat = xhat.view(-1, 96, 96, 3)

# label_hat = alignnet(causal_latent)
#%%