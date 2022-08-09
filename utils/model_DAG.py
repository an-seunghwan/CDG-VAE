#%%
import torch
from torch import nn
#%% 
#Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
#This program is free software; 
#you can redistribute it and/or modify
#it under the terms of the MIT License.
#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the MIT License for more details.

class MaskLayer(nn.Module):
    def __init__(self, concept=4):
        super().__init__()
        self.concept = concept

        self.elu = nn.ELU()
        self.net1 = nn.Sequential(
        nn.Linear(1 , 32),
        nn.ELU(),
        nn.Linear(32, 1),
        )
        self.net2 = nn.Sequential(
        nn.Linear(1 , 32),
        nn.ELU(),
        nn.Linear(32, 1),
        )
        self.net3 = nn.Sequential(
        nn.Linear(1 , 32),
        nn.ELU(),
        nn.Linear(32, 1),
        )
        self.net4 = nn.Sequential(
        nn.Linear(1 , 32),
        nn.ELU(),
        nn.Linear(32, 1)
        )

    def forward(self, z):
        z = z.view(-1, self.concept, 1)
        z1, z2, z3, z4= z[:, 0], z[:, 1], z[:, 2], z[:, 3]

        rx1 = self.net1(z1)
        rx2 = self.net2(z2)
        rx3 = self.net3(z3)
        rx4 = self.net4(z4)

        h = torch.cat((rx1,rx2,rx3,rx4), dim=1)

        return h
#%%
# model = MaskLayer()
# x = torch.randn(10, 4)
# model(x)
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
                #  s_act='tanh',
                #  t_act='relu'):
        super(CouplingLayer, self).__init__()

        # activations = {'relu': nn.ReLU, 'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh}
        # s_act_func = activations[s_act]
        # t_act_func = activations[t_act]

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
        
        # self.scale_net = nn.Sequential(
        #     nn.Linear(input_dim, hidden_dim), 
        #     s_act_func(),
        #     nn.Linear(hidden_dim, hidden_dim), 
        #     s_act_func(),
        #     nn.Linear(hidden_dim, input_dim)).to(device)
        # self.translate_net = nn.Sequential(
        #     nn.Linear(input_dim, hidden_dim), 
        #     t_act_func(),
        #     nn.Linear(hidden_dim, hidden_dim), 
        #     t_act_func(),
        #     nn.Linear(hidden_dim, input_dim)).to(device)

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
            CouplingLayer(config["replicate"], device, reverse=False, hidden_dim=config["hidden_dim"]).to(device),
            CouplingLayer(config["replicate"], device, reverse=True, hidden_dim=config["hidden_dim"]).to(device),
            CouplingLayer(config["replicate"], device, reverse=False, hidden_dim=config["hidden_dim"]).to(device)
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
    
# class INN(nn.Module):
#     def __init__(self,
#                  config,
#                  device='cpu'):
#         super(INN, self).__init__()

#         self.coupling_layer = [
#             CouplingLayer(config["replicate"], device, reverse=False, hidden_dim=config["hidden_dim"]).to(device),
#             CouplingLayer(config["replicate"], device, reverse=True, hidden_dim=config["hidden_dim"]).to(device),
#             CouplingLayer(config["replicate"], device, reverse=False, hidden_dim=config["hidden_dim"]).to(device)
#         ]

#     def forward(self, inputs, inverse=False):
#         h = inputs
#         if inverse:
#             for i, c_layer in enumerate(reversed(self.coupling_layer)):
#                 h = c_layer.inverse(h)
#         else:
#             for i, c_layer in enumerate(self.coupling_layer):
#                 h = c_layer(h)
#         return h
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

# class GeneralizedLinearSEM(nn.Module):
#     def __init__(self,
#                  config,
#                  device='cpu'):
#         super(GeneralizedLinearSEM, self).__init__()

#         self.config = config

#         self.inn = [INN(config, device) for _ in range(config["node"])]
        
#     def forward(self, inputs, inverse=False):
#         inputs_transformed = []
#         for i, layer in enumerate(self.inn):
#             inputs_transformed.append(layer(inputs[i], inverse))
#         inputs_transformed = torch.stack(inputs_transformed, dim=1).permute(0, 2, 1).contiguous()
#         return inputs_transformed
#%%
def main():
    config = {
        "n": 10,
        "hidden_dim": 8,
        "node": 4,
        "replicate": 2,
    }

    label = torch.randn(config["n"], config["node"])
    label = label.unsqueeze(dim=2)
    label = label.repeat(1, 1, config["replicate"])
    label = [x.squeeze(dim=1).contiguous() for x in torch.split(label, 1, dim=1)]

    # inn = [INN(config) for _ in range(config["node"])]

    # label_transformed = []
    # for i, c_layer in enumerate(inn):
    #     label_transformed.append(inn[i](label[i], inverse=True))

    # label_ = []
    # for i, c_layer in enumerate(inn):
    #     label_.append(inn[i](label_transformed[i], inverse=False))
    
    model = GeneralizedLinearSEM(config)
    # forward
    label_transformed = model(label, inverse=False)
    label_transformed_ = [x.squeeze(dim=2) for x in torch.split(label_transformed, 1, dim=2)]
    # inverse
    label_ = model(label_transformed_, inverse=True)
    label_ = [x.squeeze(dim=2) for x in torch.split(label_, 1, dim=2)]
    
    """
    Generalized Linear SEM:
    u = g((I - B^T)^-1 * epsilon)
    g^-1(u) = B^T * g^-1(u) + epsilon
    """
    B = torch.randn(config["node"], config["node"]) / 10 + torch.eye(config["node"])
    
    recon = torch.pow(label_transformed - torch.matmul(label_transformed, B), 2).sum()
    
    assert torch.allclose(torch.stack(label), torch.stack(label_))
    
    print('model pass test!')
#%%
if __name__ == '__main__':
    main()
#%%