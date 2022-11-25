#%%
import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.chdir(os.path.dirname(os.path.abspath(__file__)))
#%%
import numpy as np
import pandas as pd
import tqdm
from PIL import Image
import matplotlib.pyplot as plt
# plt.switch_backend('agg')

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
    TabularDataset, 
    TestTabularDataset,
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
    parser.add_argument('--model', type=str, default='GAM', 
                        help='VAE based model options: VAE, InfoMax, GAM')

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
    parser.add_argument('--input_dim', default=5, type=int,
                        help='input dimension')
    
    # optimization options
    parser.add_argument('--epochs', default=200, type=int,
                        help='maximum iteration')
    parser.add_argument('--batch_size', default=256, type=int,
                        help='batch size')
    parser.add_argument('--lr', default=0.01, type=float,
                        help='learning rate')
    parser.add_argument('--lr_D', default=0.001, type=float, # InfoMax
                        help='learning rate for discriminator in InfoMax')
    
    # loss coefficients
    parser.add_argument('--beta', default=0.01, type=float,
                        help='observation noise')
    parser.add_argument('--lambda', default=1, type=float,
                        help='weight of label alignment loss')
    parser.add_argument('--gamma', default=1, type=float, # InfoMax
                        help='weight of f-divergence (lower bound of information)')
    
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
    dataset = TabularDataset(config)
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)
    
    from causallearn.search.ConstraintBased.PC import pc
    from causallearn.utils.GraphUtils import GraphUtils
    
    cg = pc(data=dataset.train.to_numpy(), 
            alpha=0.05, 
            indep_test='chisq') 
    print(cg.G)
    
    # visualization
    pdy = GraphUtils.to_pydot(cg.G, labels=dataset.continuous)
    pdy.write_png('./assets/loan/dag_train_loan.png')
    fig = Image.open('./assets/loan/dag_train_loan.png')
    wandb.log({'Baseline DAG (Train)': wandb.Image(fig)})
    #%%
    """
    Causal Adjacency Matrix
    [Mortgage, Income] -> CCAvg
    [Experience, Age] -> CCAvg
    """
    B = torch.zeros(config["node"], config["node"])
    B[:-1, -1] = 1
    
    """adjacency matrix scaling"""
    if config["adjacency_scaling"]:
        indegree = B.sum(axis=0)
        mask = (indegree != 0)
        B[:, mask] = B[:, mask] / indegree[mask]
    print(B)
    #%%
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
        mask = [2, 2, 1]
        from modules.model import GAM
        model = GAM(B, mask, config, device) 
    
    else:
        raise ValueError('Not supported model!')
        
    model = model.to(device)
    
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config["lr"]
    )
    #%%
    model.train()
    
    for epoch in range(config["epochs"]):
        if config["model"] == 'VAE':
            logs = train_VAE(dataset, dataloader, model, config, optimizer, device)
        elif config["model"] == 'InfoMax':
            logs = train_InfoMax(dataset, dataloader, model, discriminator, config, optimizer, optimizer_D, device)
        elif config["model"] == 'GAM':
            logs = train_GAM(dataset, dataloader, model, config, optimizer, device)
        else:
            raise ValueError('Not supported model!')
        
        print_input = "[epoch {:03d}]".format(epoch + 1)
        print_input += ''.join([', {}: {:.4f}'.format(x, np.mean(y)) for x, y in logs.items()])
        print(print_input)
        
        """update log"""
        wandb.log({x : np.mean(y) for x, y in logs.items()})
    #%%
    testdataset = TestTabularDataset(config)
    testdataloader = DataLoader(testdataset, batch_size=config["batch_size"], shuffle=True)
    #%%
    train_recon = []
    for (x_batch, y_batch) in tqdm.tqdm(iter(dataloader), desc="inner loop"):
        if config["cuda"]:
            x_batch = x_batch.cuda()
            y_batch = y_batch.cuda()
        
        with torch.no_grad():
            out = model(x_batch, deterministic=True)
        train_recon.append(out[-1])
    train_recon = torch.cat(train_recon, dim=0)
    #%%
    test_recon = []
    for (x_batch, y_batch) in tqdm.tqdm(iter(testdataloader), desc="inner loop"):
        if config["cuda"]:
            x_batch = x_batch.cuda()
            y_batch = y_batch.cuda()
        
        with torch.no_grad():
            out = model(x_batch, deterministic=True)
        test_recon.append(out[-1])
    test_recon = torch.cat(test_recon, dim=0)
    #%%
    torch.manual_seed(config["seed"])
    randn = torch.randn(5000, config["node"])
    with torch.no_grad():
        _, latent, _ = model.transform(randn, log_determinant=False)
        if config["model"] == 'GAM':
            sample_recon = model.decode(latent)[1]
        else:
            sample_recon = model.decoder(torch.cat(latent, dim=1))
    #%%
    """PC algorithm : CPDAG"""
    cols = [item for sublist in dataset.topology for item in sublist]
    train_df = pd.DataFrame(train_recon.numpy(), columns=cols)
    train_df = train_df[dataset.continuous]
    
    cg = pc(data=train_df.to_numpy(), 
            alpha=0.05, 
            indep_test='fisherz') 
    print(cg.G)
    
    # visualization
    pdy = GraphUtils.to_pydot(cg.G, labels=train_df.columns)
    pdy.write_png('./assets/loan/dag_recon_train_loan.png')
    fig = Image.open('./assets/loan/dag_recon_train_loan.png')
    wandb.log({'Reconstructed DAG (Train)': wandb.Image(fig)})
    #%%
    cols = [item for sublist in dataset.topology for item in sublist]
    test_df = pd.DataFrame(test_recon.numpy(), columns=cols)
    test_df = test_df[dataset.continuous]
    
    cg = pc(data=test_df.to_numpy(), 
            alpha=0.05, 
            indep_test='fisherz') 
    print(cg.G)
    
    # visualization
    pdy = GraphUtils.to_pydot(cg.G, labels=test_df.columns)
    pdy.write_png('./assets/loan/dag_recon_test_loan.png')
    fig = Image.open('./assets/loan/dag_recon_test_loan.png')
    wandb.log({'Reconstructed DAG (Test)': wandb.Image(fig)})
    #%%
    cols = [item for sublist in dataset.topology for item in sublist]
    sample_df = pd.DataFrame(sample_recon.numpy(), columns=cols)
    sample_df = sample_df[dataset.continuous]
    
    cg = pc(data=sample_df.to_numpy(), 
            alpha=0.05, 
            indep_test='fisherz') 
    print(cg.G)
    
    # visualization
    pdy = GraphUtils.to_pydot(cg.G, labels=sample_df.columns)
    pdy.write_png('./assets/loan/dag_recon_sample_loan.png')
    fig = Image.open('./assets/loan/dag_recon_sample_loan.png')
    wandb.log({'Reconstructed DAG (Sampled)': wandb.Image(fig)})
    #%%
    # """model save"""
    # torch.save(model.state_dict(), './assets/model_{}_{}.pth'.format(config["model"], config["scm"]))
    # artifact = wandb.Artifact('model_{}_{}'.format(config["model"], config["scm"]), 
    #                         type='model',
    #                         metadata=config) # description=""
    # artifact.add_file('./assets/model_{}_{}.pth'.format(config["model"], config["scm"]))
    # artifact.add_file('./main.py')
    # artifact.add_file('./modules/model.py')
    # wandb.log_artifact(artifact)
    #%%
    wandb.run.finish()
#%%
if __name__ == '__main__':
    main()
#%%