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
def checkerboard_mask(width, reverse=False, device='cpu'):
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
    mask = torch.tensor(checkerboard, requires_grad=False).to(device)
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
                 device='cpu',
                 reverse=False,
                 hidden_dim=8,):
        super(CouplingLayer, self).__init__()

        self.mask = checkerboard_mask(input_dim, reverse=reverse).to(device)

        self.scale_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), 
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), 
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Tanh()).to(device)
        self.translate_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), 
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), 
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)).to(device)
        
    def inverse(self, inputs):
        u = inputs * self.mask
        u += (1 - self.mask) * (inputs - self.translate_net(self.mask * inputs)) * torch.exp(-self.scale_net(self.mask * inputs))
        return u
        
    def forward(self, inputs):
        z = self.mask * inputs
        z += (1 - self.mask) * (inputs * torch.exp(self.scale_net(self.mask * inputs)) + self.translate_net(self.mask * inputs))
        return z
#%%
class INN(nn.Module):
    def __init__(self,
                 config,
                 device='cpu'):
        super(INN, self).__init__()

        self.coupling_layer = [
            CouplingLayer(config["embedding_dim"], device, reverse=False, hidden_dim=config["hidden_dim"]).to(device),
            CouplingLayer(config["embedding_dim"], device, reverse=True, hidden_dim=config["hidden_dim"]).to(device),
            CouplingLayer(config["embedding_dim"], device, reverse=False, hidden_dim=config["hidden_dim"]).to(device)
        ]
        
        self.coupling_module = nn.Sequential(*self.coupling_layer)

    def inverse(self, inputs):
        h = inputs
        for c_layer in reversed(self.coupling_layer):
                h = c_layer.inverse(h)
        return h
        
    def forward(self, inputs):
        h = self.coupling_module(inputs)
        return h
#%%
class GeneralizedLinearSEM(nn.Module):
    def __init__(self,
                 config,
                 device='cpu'):
        super(GeneralizedLinearSEM, self).__init__()

        self.config = config

        self.inn = [INN(config, device) for _ in range(config["node"])]
    
    def forward(self, inputs, inverse=False):
        if inverse:
            inputs_transformed = list(map(lambda x, layer: layer.inverse(x), inputs, self.inn))
        else:
            inputs_transformed = list(map(lambda x, layer: layer(x), inputs, self.inn))
        inputs_transformed = torch.stack(inputs_transformed, dim=1).permute(0, 2, 1).contiguous()
        return inputs_transformed
#%%
class AlignNet(nn.Module):
    def __init__(self, config, device, output_dim=1, hidden_dim=4):
        super(AlignNet, self).__init__()
        
        self.config = config
        self.device = device
        
        self.net = [
            nn.Sequential(
                nn.Linear(config["embedding_dim"], output_dim),
                # nn.ReLU(),
                # nn.Linear(hidden_dim, output_dim)
                ).to(device) 
            for _ in range(config["node"])]
        
    def forward(self, input):
        # split_latent = list(map(lambda x: x.squeeze(dim=2), torch.split(input, 1, dim=2)))
        h = list(map(lambda x, layer: layer(x), input, self.net))
        h = torch.cat(h, dim=1)
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
            nn.Linear(300, config["node"] * config["embedding_dim"]),
            nn.BatchNorm1d(config["node"] * config["embedding_dim"]),
        ).to(device)
        
        """Build Vector Quantizer"""
        self.vq_layer = [VectorQuantizer(config, device).to(device)
                       for _ in range(config["node"])]
        
        self.B = B.to(device) # weighted adjacency matrix
        # self.batchnorm = nn.BatchNorm1d(config["node"] * config["embedding_dim"]).to(device)
        
        """Generalized Linear SEM"""
        self.inn = [INN(config, device) for _ in range(config["node"])]
        
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
        
        """alignment net"""
        self.alignnet = AlignNet(config, device)
        
        # u = 0.5 * torch.log((1. + u) / (1. - u) + 1e-8) # tanh inverse function
        
    def forward(self, input):
        latent = self.encoder(nn.Flatten()(input)) # [batch, node*embedding_dim]
        
        """Vector Quantization w.r.t. exogenous variables"""
        latent = latent.view(-1, self.config["node"], self.config["embedding_dim"]).contiguous() # [batch, node, embedding_dim]
        latent = [x.squeeze(dim=1) for x in torch.split(latent, 1, dim=1)]
        vq_outputs = list(map(lambda x, layer: layer(x), latent, self.vq_layer))
        
        vq_loss = sum([x[1] for x in vq_outputs])
        quantized_latent = [x[0] for x in vq_outputs]
        quantized_latent = torch.stack(quantized_latent, dim=2)

        """Latent Generating Process"""
        causal_latent = torch.matmul(quantized_latent, torch.inverse(self.I - self.B)) # [batch, embedding_dim, node]
        causal_latent_orig = causal_latent.clone() # save for DAG reconstruction loss (before transform)
        causal_latent = [x.squeeze(dim=2) for x in torch.split(causal_latent, 1, dim=2)]
        causal_latent = list(map(lambda x, layer: torch.tanh(layer(x)), causal_latent, self.inn)) # intervention range (-1, 1)
        # causal_latent_inv = list(map(lambda x, layer: layer.inverse(x), causal_latent, self.inn))

        xhat = self.decoder(torch.cat(causal_latent, dim=1).contiguous())
        xhat = xhat.view(-1, 96, 96, 3)
        
        label_hat = self.alignnet(causal_latent)
        
        return causal_latent_orig, causal_latent, xhat, vq_loss, label_hat
#%%
def main():
    config = {
        "n": 10,
        "num_embeddings": 10,
        "node": 4,
        "embedding_dim": 2,
        "beta": 0.25,
        "hidden_dim": 4, 
    }
    
    B = torch.randn(config["node"], config["node"]) / 10 + torch.eye(config["node"])
    model = VAE(B, config, 'cpu')
    for x in model.parameters():
        print(x.shape)
        
    batch = torch.rand(config["n"], 96, 96, 3)
    causal_latent_orig, causal_latent, xhat, vq_loss, label_hat = model(batch)
    
    assert causal_latent_orig.shape == (config["n"], config["embedding_dim"], config["node"])
    assert causal_latent[0].shape == (config["n"], config["embedding_dim"])
    assert xhat.shape == (config["n"], 96, 96, 3)
    assert label_hat.shape == (config["n"], config["node"])
    
    causal_latent_inv = [0.5 * torch.log((1. + x) / (1. - x) + 1e-8) for x in causal_latent] # tanh inverse function
    causal_latent_inv = list(map(lambda x, layer: layer.inverse(x), causal_latent_inv, model.inn))
    
    assert torch.isclose(causal_latent_orig.view(-1,  config["embedding_dim"] * config["node"]), 
                        torch.cat(causal_latent_inv, dim=1)).sum()
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
#     "embedding_dim": 2,
#     "beta": 0.25,
#     "hidden_dim": 4, 
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

# inn = [INN(config, device) for _ in range(config["node"])]

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
# alignnet = AlignNet(config, device)

# latent = encoder(nn.Flatten()(x)) # [batch, node*embedding_dim]

# """Vector Quantization w.r.t. exogenous variables"""
# latent = latent.view(-1, config["node"], config["embedding_dim"]).contiguous()
# latent = [x.squeeze(dim=1) for x in torch.split(latent, 1, dim=1)]
# vq_outputs = list(map(lambda x, layer: layer(x), latent, vq_layer))

# vq_loss = sum([x[1] for x in vq_outputs])
# quantized_latent = [x[0] for x in vq_outputs]
# quantized_latent = torch.stack(quantized_latent, dim=2)

# """Latent Generating Process"""
# causal_latent = torch.matmul(quantized_latent, torch.inverse(I - B)) # [batch, embedding_dim, node]
# causal_latent_orig = causal_latent.clone() # save for DAG reconstruction loss (before transform)
# causal_latent = [x.squeeze(dim=2) for x in torch.split(causal_latent, 1, dim=2)]
# causal_latent = list(map(lambda x, layer: torch.tanh(layer(x)), causal_latent, inn)) # intervention range (-1, 1)
# # causal_latent_inv = list(map(lambda x, layer: layer.inverse(x), causal_latent, inn))

# xhat = decoder(torch.cat(causal_latent, dim=1).contiguous())
# xhat = xhat.view(-1, 96, 96, 3)

# label_hat = alignnet(causal_latent)
#%%