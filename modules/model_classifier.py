#%%
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
#%%
class Classifier(nn.Module):
    def __init__(self, mask, config, device):
        super(Classifier, self).__init__()
        
        self.mask = mask
        self.config = config
        self.device = device
        
        self.classify = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(3*config["image_size"]*config["image_size"], 300),
                nn.ELU(),
                nn.Linear(300, 300),
                nn.ELU(),
                nn.Linear(300, 1),
            ).to(device) for _ in range(config["node"])])
        
    def forward(self, input):
        out = [C(nn.Flatten()(input * m.to(self.device))) for C, m in zip(self.classify, self.mask)]
        return torch.cat(out, dim=-1)
#%%
def main():
    config = {
        "dataset": 'pendulum',
        "image_size": 64,
        "n": 10,
        "node": 4,
    }
    
    if config["dataset"] == 'pendulum':
        """masking"""
        mask = []
        # light
        m = torch.zeros(config["image_size"], config["image_size"], 3)
        m[:20, ...] = 1
        mask.append(m)
        # angle
        m = torch.zeros(config["image_size"], config["image_size"], 3)
        m[20:51, ...] = 1
        mask.append(m)
        # shadow
        m = torch.zeros(config["image_size"], config["image_size"], 3)
        m[51:, ...] = 1
        mask.append(m)
        m = torch.zeros(config["image_size"], config["image_size"], 3)
        m[51:, ...] = 1
        mask.append(m)
    elif config["dataset"] == 'celeba':
        raise NotImplementedError('Not yet for CELEBA dataset')
    
    model = Classifier(mask, config, 'cpu')
    for x in model.parameters():
        print(x.shape)
    batch = torch.rand(config["n"], config["image_size"], config["image_size"], 3)
    
    pred = model(batch)
    
    assert pred.shape == (config["n"], config["node"])
    
    print("Model test pass!")
#%%
if __name__ == '__main__':
    main()
#%%