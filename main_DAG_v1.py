#%%
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.chdir(os.path.dirname(os.path.abspath(__file__)))
#%%
import numpy as np
import pandas as pd
import tqdm
from PIL import Image
import io
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
plt.switch_backend('agg')

import torch
from torch import nn
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
    tags=["true_DAG"]
)
#%%
import argparse
def get_args(debug):
    parser = argparse.ArgumentParser('parameters')

    parser.add_argument('--seed', type=int, default=5, 
                        help='seed for repeatable results')

    parser.add_argument('--node', default=4, type=int,
                        help='the number of nodes')
    parser.add_argument('--batch_size', default=256, type=int,
                        help='batch size')
    parser.add_argument("--hidden_dim", default=4, type=int,
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
    parser.add_argument('--w_threshold', default=0.5, type=float,
                        help='weight adjacency matrix threshold')
    parser.add_argument('--lambda', default=0.1, type=float,
                        help='weight of LASSO regularization')
    parser.add_argument('--progress_rate', default=0.1, type=float,
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
def main():
    config = vars(get_args(debug=True)) # default configuration
    # config["cuda"] = torch.cuda.is_available()
    # device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    config["cuda"] = False
    device = torch.device('cpu')
    wandb.config.update(config)

    set_random_seed(config["seed"])
    torch.manual_seed(config["seed"])
    if config["cuda"]:
        torch.cuda.manual_seed(config["seed"])

    train_imgs = os.listdir('./utils/causal_data/pendulum/train')
    label = np.array([x[:-4].split('_')[1:] for x in train_imgs]).astype(float)
    label = label - label.mean(axis=0).round(2) 
    print("Label Dataset Mean:", label.mean(axis=0).round(2))
    print("Label Dataset Std:", label.std(axis=0).round(2))
    # label = label / label.std(axis=0)
    names = ['light', 'theta', 'length', 'position']
    
    data = label[np.random.choice(len(label), int(len(label) * 0.1), replace=False), :]

    """PC algorithm"""
    from causallearn.search.ConstraintBased.PC import pc
    cg = pc(data=data, 
            alpha=0.05, 
            indep_test='fisherz')
    print(cg.G)
    # visualization
    pdy = GraphUtils.to_pydot(cg.G, labels=names)
    pdy.write_png('graph_PC.png')
    Image.open('graph_PC.png')

    """FCI algorithm"""
    from causallearn.search.ConstraintBased.FCI import fci
    from causallearn.utils.GraphUtils import GraphUtils
    
    G, edges = fci(dataset=data, 
                   independence_test_method='fisherz', 
                   alpha=0.05)
    print(G) # https://www.nature.com/articles/s41598-020-59669-x
    print(G.graph)
    # visualization
    pdy = GraphUtils.to_pydot(G, labels=names)
    pdy.write_png('graph_FCI.png')
    Image.open('graph_FCI.png')
    
    # """ground-truth"""
    # B_true = np.zeros((config["node"], config["node"]))
    # B_true[:2, 2:] = 1
    # B_true = B_true.astype(float)
    
    # B_ = B_masked.detach().cpu().numpy()
    # # thresholding
    # B_ = B_ / np.max(np.abs(B_)) # normalize weighted adjacency matrix
    # B_[np.abs(B_) < config["w_threshold"]] = 0.
    # B_ = B_.astype(float).round(2)
    # wandb.run.summary['Is DAG?'] = is_dag(B_)
    # fig = viz_heatmap(np.flipud(B_), size=(5, 4))
    # wandb.log({'heatmap_est': wandb.Image(fig)})
    
    # """accuracy"""
    # B_bin = (B_ != 0).astype(float)
    # acc = count_accuracy(B_true, B_bin)
    # wandb.log(acc)

    wandb.run.finish()
#%%
if __name__ == '__main__':
    main()
#%%