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
from torch.utils.data import TensorDataset, DataLoader

from utils.model_DAG import MaskLayer

from utils.simulation import (
    set_random_seed,
    is_dag,
)

from utils.viz import (
    viz_graph,
    viz_heatmap,
)

from utils.trac_exp import trace_expm
#%%
"""
wandb artifact cache cleanup "1GB"
"""
import sys
import subprocess
try:
    import wandb
except:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "wandb"])
    with open("../wandb_api.txt", "r") as f:
        key = f.readlines()
    subprocess.run(["wandb", "login"], input=key[0], encoding='utf-8')
    import wandb

wandb.init(
    project="(causal)VAE", 
    entity="anseunghwan",
    tags=["true_DAG"]
)
#%%
import argparse
def get_args(debug):
    parser = argparse.ArgumentParser('parameters')

    parser.add_argument('--seed', type=int, default=1, 
                        help='seed for repeatable results')

    parser.add_argument('--d', default=4, type=int,
                        help='the number of nodes')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='batch size')
    
    parser.add_argument('--rho', default=1, type=float,
                        help='rho')
    parser.add_argument('--alpha', default=3, type=float,
                        help='alpha')
    # parser.add_argument('--h', default=np.inf, type=float,
    #                     help='h')
    
    parser.add_argument('--lr', default=0.001, type=float,
                        help='learning rate')
    # parser.add_argument('--max_iter', default=100, type=int,
    #                     help='maximum iteration')
    parser.add_argument('--epochs', default=100, type=int,
                        help='maximum iteration')
    # parser.add_argument('--h_tol', default=1e-8, type=float,
    #                     help='h value tolerance')
    # parser.add_argument('--w_threshold', default=0.3, type=float,
    #                     help='weight adjacency matrix threshold')
    # parser.add_argument('--lambda', default=0.1, type=float,
    #                     help='weight of LASSO regularization')
    # parser.add_argument('--progress_rate', default=0.25, type=float,
    #                     help='progress rate')
    # parser.add_argument('--rho_max', default=1e+16, type=float,
    #                     help='rho max')
    # parser.add_argument('--rho_rate', default=2, type=float,
    #                     help='rho rate')

    if debug:
        return parser.parse_args(args=[])
    else:    
        return parser.parse_args()
#%%
def h_fun(W):
    """Evaluate DAGness constraint"""
    h = trace_expm(W * W) - W.shape[0]
    return h

def train(dataloader, model, B_est, config, optimizer):
    logs = {
        'loss': [], 
        'recon': [],
        # 'L1': [],
        'aug': [],
    }
    
    # for batch in tqdm.tqdm(iter(dataloader), desc="inner loop"):
    for batch in iter(dataloader):
        
        batch = batch[0]
        
        """Evaluate value and gradient of augmented Lagrangian."""
        loss_ = []
        
        mse = torch.nn.MSELoss()
        recon = mse(model(torch.matmul(batch, B_est)), batch)
        loss_.append(('recon', recon))
        
        # L1 = config["lambda"] * torch.norm(B_est, p=1)
        # loss_.append(('L1', L1))
        
        h = h_fun(B_est)
        aug = config["alpha"] * h
        aug += 0.5 * config["rho"] * (h ** 2)
        loss_.append(('aug', aug))
            
        loss = sum([y for _, y in loss_])
        loss_.append(('loss', loss))
        
        loss.backward()
        optimizer.step()
        
        """accumulate losses"""
        for x, y in loss_:
            logs[x] = logs.get(x) + [y.item()]
    
    return logs, B_est
#%%
def main():
    config = vars(get_args(debug=True)) # default configuration
    config["cuda"] = torch.cuda.is_available()
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    wandb.config.update(config)
    
    set_random_seed(config["seed"])
    torch.manual_seed(config["seed"])
    if config["cuda"]:
        torch.cuda.manual_seed(config["seed"])

    train_imgs = os.listdir('./utils/causal_data/pendulum/train')
    label = np.array([x[:-4].split('_')[1:] for x in train_imgs]).astype(float)
    # label = label - label.mean(axis=0) # logit
    # label = label / label.std(axis=0)
    
    label = torch.Tensor(label) 
    dataset = TensorDataset(label) 
    dataloader = DataLoader(dataset, 
                            batch_size=config["batch_size"],
                            shuffle=True)
    
    model = MaskLayer() # nonlinear transformation
    B_est = torch.zeros((config["d"], config["d"]), 
                        requires_grad=True)

    optimizer = torch.optim.Adam(list(model.parameters()) + [B_est], lr=config["lr"])
    model.train()
    
    # for epoch in tqdm.tqdm(range(config["epochs"]), desc="Finding True DAG..."):
    for epoch in range(config["epochs"]):
        logs, B_est = train(dataloader, model, B_est, config, optimizer)
        
        print_input = "[EPOCH {:03d}]".format(epoch + 1)
        print_input += ''.join([', {}: {:.4f}'.format(x, np.mean(y).round(2)) for x, y in logs.items()])
        print(print_input)
        
        """update log"""
        if epoch % 10 == 0:
            wandb.log({x : np.mean(y) for x, y in logs.items()})
        
        # """stopping rule"""
        # if h <= config["h_tol"] or rho >= config["rho_max"]:
        #     break
    
    B_est
    
    wandb.run.finish()
#%%
if __name__ == '__main__':
    main()
#%%