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

from modules.model import TVAE

from modules.train import train_TVAE
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
    tags=["Tabular", "VAEBased"],
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
    parser.add_argument('--dataset', type=str, default='adult', 
                        help='Dataset options: loan, adult, covtype')
    parser.add_argument('--model', type=str, default='TVAE', 
                        help='VAE based model option: TVAE')

    # causal structure
    parser.add_argument("--node", default=3, type=int,
                        help="the number of nodes")
    parser.add_argument("--scm", default='linear', type=str,
                        help="SCM structure options: linear or nonlinear")
    parser.add_argument("--flow_num", default=1, type=int,
                        help="the number of invertible NN flow")
    parser.add_argument("--inverse_loop", default=100, type=int,
                        help="the number of inverse loop")
    parser.add_argument("--factor", default=[1, 1, 1], type=arg_as_list, 
                        help="Numbers of latents allocated to each factor in image")
    
    # parser.add_argument("--label_normalization", default=False, type=bool,
    #                     help="If True, normalize additional information label data")
    parser.add_argument("--adjacency_scaling", default=True, type=bool,
                        help="If True, scaling adjacency matrix with in-degree")
    
    # optimization options
    parser.add_argument('--epochs', default=300, type=int,
                        help='maximum iteration')
    parser.add_argument('--batch_size', default=256, type=int,
                        help='batch size')
    parser.add_argument('--lr', default=0.001, type=float,
                        help='learning rate')
    parser.add_argument('--weight_decay', default=1e-5, type=float, 
                        help='weight decay parameter')
    
    # loss coefficients
    # parser.add_argument('--beta', default=0.01, type=float,
    #                     help='observation noise')
    parser.add_argument('--lambda', default=5, type=float,
                        help='weight of label alignment loss')
    
    if debug:
        return parser.parse_args(args=[])
    else:    
        return parser.parse_args()
#%%
def main():
    #%%
    config = vars(get_args(debug=False)) # default configuration
    config["cuda"] = torch.cuda.is_available()
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    wandb.config.update(config)
    
    set_random_seed(config["seed"])
    torch.manual_seed(config["seed"])
    if config["cuda"]:
        torch.cuda.manual_seed(config["seed"])
    #%%
    """dataset"""
    import importlib
    dataset_module = importlib.import_module('modules.{}_datasets'.format(config["dataset"]))
    TabularDataset2 = dataset_module.TabularDataset2
    
    dataset = TabularDataset2(config)
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)
    
    config["input_dim"] = dataset.transformer.output_dimensions
    #%%
    """
    Causal Adjacency Matrix
    Personal Loan:
        [Mortgage, Income] -> CCAvg
        [Experience, Age] -> CCAvg
    Adult:
        capital-gain -> [income, educational-num, hours-per-week]
        capital-loss -> [income, educational-num, hours-per-week]
    Forest Cover Type Prediction:
    """
    if config["dataset"] == 'loan':
        B = torch.zeros(config["node"], config["node"])
        B[:-1, -1] = 1
    
    elif config["dataset"] == 'adult':
        B = torch.zeros(config["node"], config["node"])
        B[:-1, -1] = 1
    
    elif config["dataset"] == 'covtype':
        B = torch.zeros(config["node"], config["node"])
        B[[0, 3, 4, 5], 1] = 1
        B[[3, 4, 5], 2] = 1
        B[[0, 5], 3] = 1
        
    else:
        raise ValueError('Not supported dataset!')
    
    """adjacency matrix scaling"""
    if config["adjacency_scaling"]:
        indegree = B.sum(axis=0)
        mask = (indegree != 0)
        B[:, mask] = B[:, mask] / indegree[mask]
    print(B)
    #%%
    """model"""
    decoder_dims = []
    for l in dataset.transformer.output_info_list:
        decoder_dims.append(sum([x.dim for x in l]))
    
    if config["dataset"] == 'loan':
        mask_ = [0, 2, 2, 1]
        mask_ = np.cumsum(mask_)
    elif config["dataset"] == 'adult':
        mask_ = [0, 1, 1, 3]
        mask_ = np.cumsum(mask_)
    elif config["dataset"] == 'covtype':
        mask_ = [0, 1, 1, 2, 1, 1, 1 + 7]
        mask_ = np.cumsum(mask_)
    else:
        raise ValueError('Not supported dataset!')
    
    mask = []
    for j in range(len(mask_) - 1):
        mask.append(sum(decoder_dims[mask_[j]:mask_[j+1]]))
    
    model = TVAE(B, mask, config, device).to(device)
    
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config["lr"],
        weight_decay=config["weight_decay"]
    )
    #%%
    model.train()
    
    for epoch in range(config["epochs"]):
        logs = train_TVAE(dataset.transformer.output_info_list, dataset, dataloader, model, config, optimizer, device)
        
        print_input = "[epoch {:03d}]".format(epoch + 1)
        print_input += ''.join([', {}: {:.4f}'.format(x, np.mean(y)) for x, y in logs.items()])
        print(print_input)
        
        """update log"""
        wandb.log({x : np.mean(y) for x, y in logs.items()})
    #%%
    """model save"""
    torch.save(model.state_dict(), './assets/tabular_{}_{}.pth'.format(config["model"], config["dataset"]))
    artifact = wandb.Artifact('tabular_{}_{}'.format(config["model"], config["dataset"]), 
                            type='model',
                            metadata=config) # description=""
    artifact.add_file('./assets/tabular_{}_{}.pth'.format(config["model"], config["dataset"]))
    artifact.add_file('./main.py')
    artifact.add_file('./modules/model.py')
    wandb.log_artifact(artifact)
    #%%
    wandb.run.finish()
#%%
if __name__ == '__main__':
    main()
#%%