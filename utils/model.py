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
def checkerboard_mask(width, reverse=False):
    """
    Reference:
    [1]: https://github.com/chrischute/real-nvp/blob/df51ad570baf681e77df4d2265c0f1eb1b5b646c/util/array_util.py#L78
    
    Get a checkerboard mask, such that no two entries adjacent entries
    have the same value. In non-reversed mask, top-left entry is 0.
    Args:
        width (int): Number of columns in the mask.
        requires_grad (bool): Whether the tensor requires gradient. (default: False)
    Returns:
        mask (torch.tensor): Checkerboard mask of shape (1, width).
    """
    checkerboard = [j % 2 for j in range(width)]
    mask = torch.tensor(checkerboard, requires_grad=False)
    if reverse:
        mask = 1 - mask
    # Reshape to (1, width) for broadcasting with tensors of shape (B, W)
    mask = mask.view(1, width)
    return mask
#%%
class CouplingLayer(nn.Module):
    """
    An implementation of a coupling layer from RealNVP (https://arxiv.org/abs/1605.08803).
    Reference:
    [1]: https://github.com/ikostrikov/pytorch-flows/blob/master/flows.py
    """
    def __init__(self,
                 input_dim,
                 reverse=False,
                 hidden_dim=64,
                 s_act='tanh',
                 t_act='relu'):
        super(CouplingLayer, self).__init__()

        activations = {'relu': nn.ReLU, 'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh}
        s_act_func = activations[s_act]
        t_act_func = activations[t_act]

        self.mask = checkerboard_mask(input_dim, reverse=reverse)

        self.scale_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), 
            s_act_func(),
            nn.Linear(hidden_dim, hidden_dim), 
            s_act_func(),
            nn.Linear(hidden_dim, input_dim))
        self.translate_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), 
            t_act_func(),
            nn.Linear(hidden_dim, hidden_dim), 
            t_act_func(),
            nn.Linear(hidden_dim, input_dim))

    def inverse(self, inputs):
        u = inputs * self.mask
        u += (1 - self.mask) * (inputs - self.translate_net(self.mask * inputs)) * torch.exp(-self.scale_net(self.mask * inputs))
        return u
        
    def forward(self, inputs):
        z = self.mask * inputs
        z += (1 - self.mask) * (inputs * torch.exp(self.scale_net(self.mask * inputs)) + self.translate_net(self.mask * inputs))
        return z
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
        ).to(device)
        
        """Build Vector Quantizer"""
        self.vq_layer = VectorQuantizer(self.config, device).to(device)
        
        self.B = B
        self.batchnorm = nn.BatchNorm1d(config["node"] * config["embedding_dim"])
        
        self.coupling_layer = [CouplingLayer(config["embedding_dim"], reverse=False),
                        CouplingLayer(config["embedding_dim"], reverse=True),
                        CouplingLayer(config["embedding_dim"], reverse=False)]
        
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
        
    def inverse(self, input):
        u = input
        for c_layer in reversed(self.coupling_layer):
            u = c_layer.inverse(u)
        return u
    
    def forward(self, input):
        latent = self.encoder(nn.Flatten()(input)) # [batch, node*embedding_dim]
        
        """Vector Quantization w.r.t. exogenous variables"""
        quantized_latent, vq_loss = self.vq_layer(latent)
        quantized_latent = quantized_latent.permute(0, 2, 1).contiguous() # [batch, embedding_dim, node]

        """Latent Generating Process"""
        causal_latent = torch.matmul(quantized_latent, torch.inverse(self.I - self.B))
        # causal_latent = self.batchnorm(causal_latent)
        causal_latent = causal_latent.permute(0, 2, 1).contiguous() # [batch, node, embedding_dim]
        causal_latent_orig = causal_latent.clone()
        for c_layer in self.coupling_layer:
            causal_latent = c_layer(causal_latent)
        causal_latent = torch.tanh(causal_latent)
        causal_latent = causal_latent.view(-1, self.config["node"] * self.config["embedding_dim"])

        xhat = self.decoder(causal_latent).view(-1, 96, 96, 3)
        return causal_latent_orig, causal_latent, xhat, vq_loss
#%%
def main():
    config = {
        "n": 10,
        "num_embeddings": 10,
        "node": 4,
        "embedding_dim": 6,
        "beta": 0.25,
    }
    
    B = torch.randn(config["node"], config["node"]) / 10 + torch.eye(config["node"])
    model = VAE(B, config, 'cpu')
    for x in model.parameters():
        print(x.shape)
        
    batch = torch.rand(config["n"], 96, 96, 3)
    causal_latent_orig, causal_latent, xhat, vq_loss = model(batch)
    
    assert causal_latent_orig.shape == (config["n"], config["node"], config["embedding_dim"])
    assert causal_latent.shape == (config["n"], config["node"] * config["embedding_dim"])
    assert xhat.shape == (config["n"], 96, 96, 3)
    
    inversed_latent = 0.5 * torch.log((1. + causal_latent) / (1. - causal_latent) + 1e-8)
    inversed_latent = model.inverse(inversed_latent.view(-1, config["node"], config["embedding_dim"]))
    
    assert torch.all(torch.isclose(causal_latent_orig, inversed_latent))
    
    print("Model test pass!")
#%%
if __name__ == '__main__':
    main()
#%%
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
# quantized_latent = quantized_latent.permute(0, 2, 1).contiguous()
#%%
# """encoder"""
# encoder = nn.Sequential(
#     nn.Linear(3*96*96, 300),
#     nn.ELU(),
#     nn.Linear(300, 300),
#     nn.ELU(),
#     nn.Linear(300, config["node"] * config["embedding_dim"]),
# )

# B = torch.randn(config["node"], config["node"]) / 10 + torch.eye(config["node"])

# x = torch.randn(10, 96, 96, 3)
# latents = encoder(nn.Flatten()(x))
#%%
# """Latent Generating Process"""
# I = torch.eye(config["node"])
# causal_latent = torch.matmul(quantized_latent, torch.inverse(I - B))
# causal_latent = causal_latent.permute(0, 2, 1).contiguous()

# coupling_layer = [CouplingLayer(config["embedding_dim"], reverse=False),
#                   CouplingLayer(config["embedding_dim"], reverse=True),
#                   CouplingLayer(config["embedding_dim"], reverse=False)]

# h1 = causal_latent
# for c_layer in coupling_layer:
#     h1 = c_layer(h1)

# h2 = h1
# for c_layer in reversed(coupling_layer):
#     h2 = c_layer.inverse(h2)

# causal_latent = causal_latent.view(-1, config["node"] * config["embedding_dim"])
#%%