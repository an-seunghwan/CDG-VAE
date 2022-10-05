#%%
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
#%%
class Classifier(nn.Module):
    def __init__(self, config, device):
        super(Classifier, self).__init__()
        
        self.config = config
        self.device = device
        
        self.classify = nn.Sequential(
            nn.Linear(config["node"], 2),
            nn.ELU(),
            nn.Linear(2, 1),
            nn.Sigmoid()
        ).to(device)
        
    def forward(self, input):
        out = self.classify(input)
        return out
#%%
def main():
    config = {
        "n": 10,
        "node": 4,
    }
    
    """Baseline Classifier"""
    model = Classifier(config, 'cpu')
    for x in model.parameters():
        print(x.shape)
    batch = torch.rand(config["n"], config["node"])
    
    pred = model(batch)
    
    assert pred.shape == (config["n"], 1)
    
    print("Downstream: Baseline Classifier pass test!")
#%%
if __name__ == '__main__':
    main()
#%%