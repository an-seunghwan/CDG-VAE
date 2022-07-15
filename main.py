#%%
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
#%%
import numpy as np
import pandas as pd
import tqdm

import torch

from utils.simulation import (
    set_random_seed,
    is_dag,
    load_data,
    count_accuracy
)

from utils.viz import (
    viz_graph,
    viz_heatmap,
)

from utils.model import (
    GAE
)

from utils.trac_exp import trace_expm
#%%
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
    project="(causal)GAE", 
    entity="anseunghwan",
    tags=["nonlinear"],
)
#%%
import argparse
def get_args(debug):
    parser = argparse.ArgumentParser('parameters')

    parser.add_argument('--seed', type=int, default=3, 
                        help='seed for repeatable results')
    parser.add_argument('--data_type', type=str, default='synthetic',
                        help='types of data: synthetis, discrete, real')
    parser.add_argument('--n', default=5000, type=int,
                        help='the number of dataset')
    parser.add_argument('--d', default=10, type=int,
                        help='the number of nodes')
    parser.add_argument('--degree', default=2, type=int,
                        help='expected number of edges')
    parser.add_argument('--graph_type', type=str, default='ER',
                        help='graph type: ER, SF, BP')
    parser.add_argument('--sem_type', type=str, default='gauss',
                        help='sem type: gauss, exp, gumbel, uniform, logistic, poisson')
    parser.add_argument('--nonlinear_type', type=str, default='nonlinear_1',
                        help='nonlinear causal structure type: nonlinear_1, nonlinear_2')

    parser.add_argument('--rho', default=1, type=float,
                        help='rho')
    parser.add_argument('--alpha', default=0, type=float,
                        help='alpha')
    parser.add_argument('--h', default=np.inf, type=float,
                        help='h')
    
    parser.add_argument("--hidden_dim", default=16, type=int,
                        help="hidden dimensions for MLP")
    parser.add_argument("--num_layer", default=2, type=int,
                        help="hidden dimensions for MLP")
    parser.add_argument("--x_dim", default=1, type=int,
                        help="dimension of each node")
    parser.add_argument("--latent_dim", default=1, type=int,
                        help="dimension of each latent node")
    
    parser.add_argument('--epochs', default=300, type=int,
                        help='maximum iteration')
    parser.add_argument('--batch_size', default=100, type=int,
                        help='batch size')
    parser.add_argument('--lr', default=0.003, type=float,
                        help='learning rate')
    parser.add_argument('--init_iter', default=5, type=int,
                        help='initial iteration')
    parser.add_argument('--early_stopping', type=bool, default=True,
                        help='if True, early stopping')
    parser.add_argument('--early_stopping_threshold', default=1.15, type=float,
                        help='early stopping threshold')
    
    parser.add_argument('--max_iter', default=20, type=int,
                        help='maximum number of iteration')
    parser.add_argument('--h_tol', default=1e-12, type=float,
                        help='h value tolerance')
    parser.add_argument('--w_threshold', default=0.2, type=float,
                        help='threshold for weighted adjacency matrix')
    parser.add_argument('--lambda', default=0, type=float,
                        help='coefficient of LASSO penalty')
    parser.add_argument('--progress_rate', default=0.25, type=float,
                        help='progress rate')
    parser.add_argument('--rho_max', default=1e+18, type=float,
                        help='maximum rho value')
    parser.add_argument('--rho_rate', default=10, type=float,
                        help='rho rate')
    
    parser.add_argument('--fig_show', default=False, type=bool)

    if debug:
        return parser.parse_args(args=[])
    else:    
        return parser.parse_args()
    
# config = {
#     "seed": 3,
#     'data_type': 'synthetic', # discrete, real
#     "n": 5000,
#     "d": 10,
#     "degree": 2,
#     "graph_type": "ER",
#     "sem_type": "gauss",
#     "nonlinear_type": "nonlinear_1",
#     "hidden_dim": 16,
#     "num_layer": 2,
#     "x_dim": 1,
#     "latent_dim": 1,
    
#     "epochs": 300,
#     "batch_size": 100,
#     "lr": 0.003,
#     "init_iter": 5,
#     "early_stopping": True,
#     "early_stopping_threshold": 1.15,
    
#     "rho": 1, # initial value
#     "alpha": 0., # initial value
#     "h": np.inf, # initial value
    
#     "max_iter": 20, 
#     "loss_type": 'l2',
#     "h_tol": 1e-12, 
#     "w_threshold": 0.2,
#     "lambda": 0.,
#     "progress_rate": 0.25,
#     "rho_max": 1e+18, 
#     "rho_rate": 10,
    
#     "fig_show": False,
# }
#%%
def h_fun(W):
    """Evaluate DAGness constraint"""
    h = trace_expm(W * W) - W.shape[0]
    return h
#%%
def train(train_loader, model, rho, alpha, config, optimizer):
    model.train()
    
    if config["cuda"]:
        X = X.cuda()
    
    logs = {
        'loss': [], 
        'recon': [],
        'L1': [],
        'aug': [],
    }
    
    for batch_num, [train_batch] in enumerate(train_loader):
        if config["cuda"]:
            train_batch = train_batch.cuda()
            
        optimizer.zero_grad()
        
        recon = model(train_batch)
        
        loss_ = []
        
        # reconstruction
        recon = 0.5 * torch.pow(recon - train_batch, 2).sum() / train_batch.size(0)
        loss_.append(('recon', recon))

        # sparsity loss
        L1 = config["lambda"] * torch.sum(torch.abs(model.W))
        loss_.append(('L1', L1))

        # augmentation and lagrangian loss
        h_A = h_fun(model.W)
        aug = 0.5 * rho * (h_A ** 2)
        aug += alpha * h_A
        loss_.append(('aug', aug))
        
        loss = sum([y for _, y in loss_])
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
    wandb.config.update(config)
    
    set_random_seed(config["seed"])
    torch.manual_seed(config["seed"])
    if config["cuda"]:
        torch.cuda.manual_seed(config["seed"])
    train_loader, W_true = load_data(config)

    wandb.run.summary['W_true'] = wandb.Table(data=pd.DataFrame(W_true))
    fig = viz_graph(W_true, size=(7, 7), show=config["fig_show"])
    wandb.log({'Graph': wandb.Image(fig)})
    fig = viz_heatmap(W_true, size=(5, 4), show=config["fig_show"])
    wandb.log({'heatmap': wandb.Image(fig)})
    
    model = GAE(config)

    if config["cuda"]:
        model.cuda()

    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config["lr"]
    )
    
    rho = config["rho"]
    alpha = config["alpha"]
    h = config["h"]
    mse_save = np.inf
    W_save = None
        
    for iteration in range(config["max_iter"]):
        
        """primal problem"""
        while rho < config["rho_max"]:
            # h_old = np.inf
            # find argmin of primal problem (local solution) = update for config["epochs"] times
            for epoch in tqdm.tqdm(range(config["epochs"]), desc="primal update"):
                logs = train(train_loader, model, rho, alpha, config, optimizer)
                
                # """FIXME"""
                # W_est = model.W.detach().data.clone()
                # h_new = h_fun(W_est).item()
                # # no change in weight estimation (convergence)
                # if abs(h_old - h_new) < 1e-12: 
                #     break
                # h_old = h_new
                
            # only one epoch is fine for finding argmin
            # logs = train(model, X, rho, alpha, config, optimizer)
            
            W_est = model.W.detach().data.clone()
            h_current = h_fun(W_est)
            if h_current.item() > config["progress_rate"] * h:
                rho *= config["rho_rate"]
            else:
                break
        
        if config["early_stopping"]:
            if np.mean(logs['recon']) / mse_save > config["early_stopping_threshold"] and h_current.item() <= 1e-7:
                W_est = W_save
                print("early stopping!")
                break
            else:
                W_save = W_est
                mse_save = np.mean(logs['recon'])
        
        """dual ascent"""
        h = h_current.item()
        alpha += rho * h_current.item()
        
        print_input = "[iteration {:03d}]".format(iteration)
        print_input += ''.join([', {}: {:.4f}'.format(x, np.mean(y).round(2)) for x, y in logs.items()])
        print_input += ', h(W): {:.8f}'.format(h)
        print(print_input)
        
        """update log"""
        wandb.log({x : np.mean(y) for x, y in logs.items()})
        wandb.log({'h(W)' : h})
        
        """stopping rule"""
        if h_current.item() <= config["h_tol"] and iteration > config["init_iter"]:
            break
    
    """final metrics"""
    W_est = W_est.numpy()
    W_est = W_est / np.max(np.abs(W_est)) # normalize weighted adjacency matrix
    W_est[np.abs(W_est) < config["w_threshold"]] = 0.
    W_est = W_est.astype(float).round(2)

    fig = viz_graph(W_est, size=(7, 7), show=config["fig_show"])
    wandb.log({'Graph_est': wandb.Image(fig)})
    fig = viz_heatmap(W_est, size=(5, 4), show=config["fig_show"])
    wandb.log({'heatmap_est': wandb.Image(fig)})

    wandb.run.summary['Is DAG?'] = is_dag(W_est)
    wandb.run.summary['W_est'] = wandb.Table(data=pd.DataFrame(W_est))
    wandb.run.summary['W_diff'] = wandb.Table(data=pd.DataFrame(W_true - W_est))

    W_ = (W_true != 0).astype(float)
    W_est_ = (W_est != 0).astype(float)
    W_diff_ = np.abs(W_ - W_est_)

    fig = viz_graph(W_diff_, size=(7, 7))
    wandb.log({'Graph_diff': wandb.Image(fig)})

    B_est = (W_est != 0).astype(float)
    B_true = (W_true != 0).astype(float)

    """accuracy"""
    acc = count_accuracy(B_true, B_est)
    wandb.log(acc)
    
    wandb.run.finish()
#%%
if __name__ == '__main__':
    main()
#%%