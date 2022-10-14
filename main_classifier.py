#%%
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.chdir(os.path.dirname(os.path.abspath(__file__)))
#%%
import numpy as np
import pandas as pd
import tqdm
from PIL import Image
import matplotlib.pyplot as plt
plt.switch_backend('agg')

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import Dataset

from modules.simulation import (
    set_random_seed,
    is_dag,
)

from modules.datasets import (
    LabeledDataset, 
)

from modules.model import (
    Classifier,
)
#%%
import sys
import subprocess
try:
    import wandb
except:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "wandb"])
    with open("./wandb_api.txt", "r") as f:
        key = f.readlines()
    subprocess.run(["wandb", "login"], input=key[0], encoding='utf-8')
    import wandb

run = wandb.init(
    project="CausalDisentangled", 
    entity="anseunghwan",
    tags=["CDMClassifier"],
)
#%%
import argparse
def get_args(debug):
    parser = argparse.ArgumentParser('parameters')
    
    parser.add_argument('--seed', type=int, default=1, 
                        help='seed for repeatable results')
    parser.add_argument("--DR", default=False, action='store_true',
                        help="If True, training model with spurious attribute")

    parser.add_argument("--node", default=4, type=int,
                        help="the number of nodes")
    parser.add_argument("--label_normalization", default=True, type=bool,
                        help="If True, normalize additional information label data")
    parser.add_argument('--image_size', default=64, type=int,
                        help='width and heigh of image')
    
    parser.add_argument('--labeled_ratio', default=1, type=float, # fully-supervised
                        help='ratio of labeled dataset for semi-supervised learning')
    
    parser.add_argument('--epochs', default=50, type=int,
                        help='maximum iteration')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='batch size')
    parser.add_argument('--lr', default=0.001, type=float,
                        help='learning rate')
    
    if debug:
        return parser.parse_args(args=[])
    else:    
        return parser.parse_args()
#%%
def train(dataloader, model, config, optimizer, device):
    logs = {
        'loss': [], 
    }
    
    for (x_batch, y_batch) in tqdm.tqdm(iter(dataloader), desc="inner loop"):
        
        if config["cuda"]:
            x_batch = x_batch.cuda()
            y_batch = y_batch.cuda()
        
        with torch.autograd.set_detect_anomaly(True):    
            optimizer.zero_grad()
            
            pred = model(x_batch)
            
            loss_ = []
            
            """Label Prediction"""
            y_hat = torch.sigmoid(pred)
            loss = F.binary_cross_entropy(y_hat, y_batch[:, :config["node"]], reduction='none').sum(axis=1).mean()
            loss_.append(('loss', loss))
            
            loss.backward()
            optimizer.step()
            
        """accumulate losses"""
        for x, y in loss_:
            logs[x] = logs.get(x) + [y.item()]
    
    return logs
#%%
def main():
    config = vars(get_args(debug=False)) # default configuration
    config["cuda"] = torch.cuda.is_available()
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    wandb.config.update(config)
    
    set_random_seed(config["seed"])
    torch.manual_seed(config["seed"])
    if config["cuda"]:
        torch.cuda.manual_seed(config["seed"])

    """dataset"""
    dataset = LabeledDataset(config)
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)
    
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
        
    model = Classifier(mask, config, device) 
    model = model.to(device)
    
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config["lr"]
    )
    
    model.train()
    
    for epoch in range(config["epochs"]):
        logs = train(dataloader, model, config, optimizer, device)
        
        print_input = "[epoch {:03d}]".format(epoch + 1)
        print_input += ''.join([', {}: {:.4f}'.format(x, np.mean(y)) for x, y in logs.items()])
        print(print_input)
        
        """update log"""
        wandb.log({x : np.mean(y) for x, y in logs.items()})
            
    """model save"""
    torch.save(model.state_dict(), './assets/CDMClassifier.pth')
    artifact = wandb.Artifact('CDMClassifier', 
                              type='model',
                              metadata=config) # description=""
    artifact.add_file('./assets/CDMClassifier.pth')
    artifact.add_file('./main_classifier.py')
    artifact.add_file('./modules/model.py')
    wandb.log_artifact(artifact)
    
    wandb.run.finish()
#%%
if __name__ == '__main__':
    main()
#%%