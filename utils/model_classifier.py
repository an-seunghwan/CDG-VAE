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
            nn.Linear(3*config["image_size"]*config["image_size"], 900),
            nn.ELU(),
            nn.Linear(900, 600),
            nn.ELU(),
            nn.Linear(600, 300),
            nn.ELU(),
            nn.Linear(300, config["node"]),
        ).to(device)
        
    def forward(self, input):
        latent = nn.Flatten()(input)
        out = self.classify(latent)
        return out
#%%
def main():
    config = {
        "image_size": 64,
        "n": 10,
        "node": 4,
    }
    
    model = Classifier(config, 'cpu')
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