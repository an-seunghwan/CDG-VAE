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
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from utils.model_DAG import GeneralizedLinearSEM

from utils.simulation import (
    set_random_seed,
    is_dag,
    count_accuracy
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
    tags=["true_DAG", "GeneralizedLinearSEM"]
)
#%%
import argparse
def get_args(debug):
    parser = argparse.ArgumentParser('parameters')

    parser.add_argument('--seed', type=int, default=1, 
                        help='seed for repeatable results')

    parser.add_argument('--node', default=4, type=int,
                        help='the number of nodes')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='batch size')
    parser.add_argument("--hidden_dim", default=8, type=int,
                        help="hidden dimensions for MLP")
    parser.add_argument("--replicate", default=2, type=int,
                        help="")
    
    parser.add_argument('--rho', default=1, type=float,
                        help='rho')
    parser.add_argument('--alpha', default=0, type=float,
                        help='alpha')
    parser.add_argument('--h', default=np.inf, type=float,
                        help='h')
    
    parser.add_argument('--lr', default=0.001, type=float,
                        help='learning rate')
    parser.add_argument('--max_iter', default=100, type=int,
                        help='maximum iteration')
    parser.add_argument('--epochs', default=100, type=int,
                        help='maximum iteration')
    parser.add_argument('--h_tol', default=1e-8, type=float,
                        help='h value tolerance')
    parser.add_argument('--w_threshold', default=0.3, type=float,
                        help='weight adjacency matrix threshold')
    parser.add_argument('--lambda', default=0.1, type=float,
                        help='weight of LASSO regularization')
    parser.add_argument('--progress_rate', default=0.25, type=float,
                        help='progress rate')
    parser.add_argument('--rho_max', default=1e+16, type=float,
                        help='rho max')
    parser.add_argument('--rho_rate', default=5, type=float,
                        help='rho rate')

    if debug:
        return parser.parse_args(args=[])
    else:    
        return parser.parse_args()
#%%
def h_fun(W):
    """Evaluate DAGness constraint"""
    h = trace_expm(W * W) - W.shape[0]
    return h

def train(dataloader, model, B_est, mask, rho, alpha, config, optimizer):
    logs = {
        'loss': [], 
        'recon': [],
        'L1': [],
        'aug': [],
    }
    
    # for batch in tqdm.tqdm(iter(dataloader), desc="inner loop"):
    for batch in iter(dataloader):
        
        batch = batch[0]
        if config["cuda"]:
            batch = batch.cuda()
        
        batch = batch.unsqueeze(dim=2)
        batch = batch.repeat(1, 1, config["replicate"])
        batch = [x.squeeze(dim=1).contiguous() for x in torch.split(batch, 1, dim=1)]
        
        loss_ = []
        
        B_masked = B_est * mask
        
        batch_transformed = model(batch, inverse=True)
        recon = 0.5 * torch.pow(batch_transformed - torch.matmul(batch_transformed, B_masked), 2).sum() / batch[0].size(0)
        loss_.append(('recon', recon))
        
        L1 = config["lambda"] * torch.norm(B_masked, p=1)
        loss_.append(('L1', L1))
        
        h = h_fun(B_masked)
        aug = alpha * h
        aug += 0.5 * rho * (h ** 2)
        loss_.append(('aug', aug))
            
        loss = sum([y for _, y in loss_])
        loss_.append(('loss', loss))
        
        loss.backward()
        optimizer.step()
        
        """accumulate losses"""
        for x, y in loss_:
            logs[x] = logs.get(x) + [y.item()]
    
    return logs, B_masked
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
    label = label - label.mean(axis=0) 
    # label = label / label.std(axis=0)

    label = torch.Tensor(label) 
    dataset = TensorDataset(label) 
    dataloader = DataLoader(dataset, 
                            batch_size=config["batch_size"],
                            shuffle=True)

    model = GeneralizedLinearSEM(config)
    B_est = torch.zeros((config["node"], config["node"]), 
                        requires_grad=True)
    mask = torch.ones(config["node"], config["node"]) - torch.eye(config["node"])

    optimizer = torch.optim.Adam(list(model.parameters()) + [B_est], lr=config["lr"])
    model.train()

    rho = config["rho"]
    alpha = config["alpha"]
    h = config["h"]
        
    for iteration in range(config["max_iter"]):
        """primal problem"""
        while rho < config["rho_max"]:
            for epoch in tqdm.tqdm(range(config["epochs"]), desc="primal update"):
                logs, B_masked = train(dataloader, model, B_est, mask, rho, alpha, config, optimizer)

            with torch.no_grad():
                h_current = h_fun(B_masked)
                if h_current.item() > config["progress_rate"] * h:
                    rho *= config["rho_rate"]
                else:
                    break
        
        """dual ascent"""
        h = h_current.item()
        alpha += rho * h_current.item()
        
        """update log"""
        print_input = "[iteration {:03d}]".format(iteration)
        print_input += ''.join([', {}: {:.4f}'.format(x, np.mean(y).round(2)) for x, y in logs.items()])
        print_input += ', h(B): {:.8f}'.format(h)
        print(print_input)
        
        wandb.log({x : np.mean(y) for x, y in logs.items()})
        wandb.log({'h(W)' : h})
        
        """stopping rule"""
        if h_current.item() <= config["h_tol"] or rho >= config["rho_max"]:
            break
    
    """ground-truth"""
    B_true = np.zeros((config["node"], config["node"]))
    B_true[:2, 2:] = 1
    B_true = B_true.astype(float)
    
    B_ = B_masked.detach().cpu().numpy()
    # thresholding
    B_ = B_ / np.max(np.abs(B_)) # normalize weighted adjacency matrix
    B_[np.abs(B_) < config["w_threshold"]] = 0.
    B_ = B_.astype(float).round(2)
    wandb.run.summary['Is DAG?'] = is_dag(B_)
    fig = viz_heatmap(np.flipud(B_), size=(5, 4))
    wandb.log({'heatmap_est': wandb.Image(fig)})
    
    B_bin = (B_ != 0).astype(float)
    """accuracy"""
    acc = count_accuracy(B_true, B_bin)
    wandb.log(acc)

    wandb.run.finish()
#%%
if __name__ == '__main__':
    main()
#%%