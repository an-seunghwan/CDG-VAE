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
    UnLabeledDataset,
)

from modules.train import (
    train_VAE,
    train_InfoMax,
    train_GAM,
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
    project="(proposal)CausalVAE", 
    entity="anseunghwan",
    tags=["DR"],
)
#%%
import argparse
import ast

def arg_as_list(s):
    v = ast.literal_eval(s)
    if type(v) is not list:
        raise argparse.ArgumentTypeError("Argument \"%s\" is not a list" % (s))
    return v

def get_args(debug):
    parser = argparse.ArgumentParser('parameters')
    
    parser.add_argument('--seed', type=int, default=1, 
                        help='seed for repeatable results')
    parser.add_argument('--model', type=str, default='GAM', 
                        help='Model options: VAE, InfoMax, GAM')

    # causal structure
    parser.add_argument("--node", default=5, type=int,
                        help="the number of nodes")
    parser.add_argument("--scm", default='linear', type=str,
                        help="SCM structure options: linear or nonlinear")
    parser.add_argument("--flow_num", default=1, type=int,
                        help="the number of invertible NN flow")
    parser.add_argument("--inverse_loop", default=100, type=int,
                        help="the number of inverse loop")
    parser.add_argument("--factor", default=[1, 1, 2], type=arg_as_list, 
                        help="Numbers of latents allocated to each factor in image")
    
    # data options
    parser.add_argument('--labeled_ratio', default=1, type=float, # fully-supervised
                        help='ratio of labeled dataset for semi-supervised learning')
    
    parser.add_argument("--label_normalization", default=True, type=bool,
                        help="If True, normalize additional information label data")
    parser.add_argument("--adjacency_scaling", default=True, type=bool,
                        help="If True, scaling adjacency matrix with in-degree")
    parser.add_argument('--image_size', default=64, type=int,
                        help='width and heigh of image')
    
    # optimization options
    parser.add_argument('--epochs', default=100, type=int,
                        help='maximum iteration')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='batch size')
    parser.add_argument('--lr', default=0.001, type=float,
                        help='learning rate')
    parser.add_argument('--lr_D', default=0.0001, type=float, # InfoMax
                        help='learning rate for discriminator in InfoMax')
    
    # loss coefficients
    parser.add_argument('--beta', default=0.1, type=float,
                        help='observation noise')
    parser.add_argument('--lambda', default=5, type=float,
                        help='weight of label alignment loss')
    parser.add_argument('--gamma', default=1, type=float, # InfoMax
                        help='weight of f-divergence (lower bound of information)')
    
    if debug:
        return parser.parse_args(args=[])
    else:    
        return parser.parse_args()
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
    
    """
    Causal Adjacency Matrix
    light -> length
    light -> position
    angle -> length
    angle -> position
    """
    B = torch.zeros(config["node"], config["node"])
    B[dataset.name.index('light'), dataset.name.index('length')] = 1
    B[dataset.name.index('light'), dataset.name.index('position')] = 1
    B[dataset.name.index('angle'), dataset.name.index('length')] = 1
    B[dataset.name.index('angle'), dataset.name.index('position')] = 1
    
    """adjacency matrix scaling"""
    if config["adjacency_scaling"]:
        indegree = B.sum(axis=0)
        mask = (indegree != 0)
        B[:, mask] = B[:, mask] / indegree[mask]
    
    """model"""
    if config["model"] == 'VAE':
        from modules.model import VAE
        model = VAE(B, config, device) 
        
    elif config["model"] == 'InfoMax':
        from modules.model import VAE, Discriminator
        model = VAE(B, config, device) 
        discriminator = Discriminator(config, device)
        discriminator = discriminator.to(device)
        
        optimizer_D = torch.optim.Adam(
            discriminator.parameters(), 
            lr=config["lr_D"]
        )
        
    elif config["model"] == 'GAM':
        """Decoder masking"""
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
        
        from modules.model import GAM
        model = GAM(B, mask, config, device) 
    
    else:
        raise ValueError('Not supported model!')
        
    model = model.to(device)
    
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config["lr"]
    )
    
    wandb.watch(model, log_freq=100) # tracking gradients
    model.train()
    
    for epoch in range(config["epochs"]):
        if config["model"] == 'VAE':
            logs, xhat = train_VAE(dataloader, model, config, optimizer, device)
        elif config["model"] == 'InfoMax':
            logs, xhat = train_InfoMax(dataloader, model, discriminator, config, optimizer, optimizer_D, device)
        elif config["model"] == 'GAM':
            logs, xhat = train_GAM(dataloader, model, config, optimizer, device)
        else:
            raise ValueError('Not supported model!')
        
        print_input = "[epoch {:03d}]".format(epoch + 1)
        print_input += ''.join([', {}: {:.4f}'.format(x, np.mean(y).round(2)) for x, y in logs.items()])
        print(print_input)
        
        """update log"""
        wandb.log({x : np.mean(y) for x, y in logs.items()})
            
        if epoch % 10 == 0:
            plt.figure(figsize=(4, 4))
            for i in range(9):
                plt.subplot(3, 3, i+1)
                plt.imshow((xhat[i].cpu().detach().numpy() + 1) / 2)
                plt.axis('off')
            plt.savefig('./assets/tmp_image_{}.png'.format(epoch))
            plt.close()
    
    """reconstruction result"""
    fig = plt.figure(figsize=(4, 4))
    for i in range(9):
        plt.subplot(3, 3, i+1)
        plt.imshow((xhat[i].cpu().detach().numpy() + 1) / 2)
        plt.axis('off')
    plt.close()
    wandb.log({'reconstruction': wandb.Image(fig)})
    
    """model save"""
    torch.save(model.state_dict(), './assets/DR_{}.pth'.format(config["model"]))
    artifact = wandb.Artifact('DR_{}'.format(config["model"]), 
                              type='model',
                              metadata=config) # description=""
    artifact.add_file('./assets/DR_{}.pth'.format(config["model"]))
    artifact.add_file('./main.py')
    artifact.add_file('./modules/model.py')
    wandb.log_artifact(artifact)
    
    wandb.run.finish()
#%%
if __name__ == '__main__':
    main()
#%%