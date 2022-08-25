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

from utils.simulation import (
    set_random_seed,
    is_dag,
)

from utils.viz import (
    viz_graph,
    viz_heatmap,
)

from utils.model_v4 import (
    VAE,
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
    project="(causal)VAE", 
    entity="anseunghwan",
    tags=["GeneralizedLinearSEM", "Identifiable"],
)
#%%
import argparse
def get_args(debug):
    parser = argparse.ArgumentParser('parameters')
    
    parser.add_argument('--seed', type=int, default=1, 
                        help='seed for repeatable results')

    parser.add_argument("--node", default=4, type=int,
                        help="the number of nodes")
    parser.add_argument("--node_dim", default=1, type=int,
                        help="dimension of each node")
    parser.add_argument("--flow_num", default=8, type=int,
                        help="the number of invertible NN (planar flow)")
    parser.add_argument("--inverse_loop", default=100, type=int,
                        help="the number of inverse loop")
    
    parser.add_argument('--epochs', default=200, type=int,
                        help='maximum iteration')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='batch size')
    parser.add_argument('--lr', default=0.001, type=float,
                        help='learning rate')
    
    parser.add_argument('--beta', default=0.1, type=float,
                        help='observation noise')
    parser.add_argument('--lambda', default=0.1, type=float,
                        help='weight of DAG reconstruction loss')
    parser.add_argument('--gamma', default=0.1, type=float,
                        help='weight of label alignment loss')
    
    parser.add_argument('--fig_show', default=False, type=bool)
    
    if debug:
        return parser.parse_args(args=[])
    else:    
        return parser.parse_args()
#%%
def train(dataloader, model, config, optimizer, device):
    logs = {
        'loss': [], 
        'recon': [],
        'DAG_recon': [],
        'KL': [],
        'align': [],
    }
    
    for (x_batch, y_batch) in tqdm.tqdm(iter(dataloader), desc="inner loop"):
        
        if config["cuda"]:
            x_batch = x_batch.cuda()
            y_batch = y_batch.cuda()
        
        # h = model.encoder(nn.Flatten()(x_batch)) # [batch, node * node_dim * 2]
        # mean, logvar = torch.split(h, model.config["node"] * model.config["node_dim"], dim=1)
        
        # """Latent Generating Process"""
        # noise = torch.randn(x_batch.size(0), model.config["node"] * model.config["node_dim"]).to(model.device) 
        # epsilon = mean + torch.exp(logvar / 2) * noise
        # epsilon = epsilon.view(-1, model.config["node_dim"], model.config["node"]).contiguous()
        # latent = torch.matmul(epsilon, torch.inverse(model.I - model.B)) # [batch, node_dim, node]
        # latent_orig = latent.clone()
        # # latent_orig.squeeze(dim=1).detach().numpy().astype(float).round(1)
        # latent = [x.squeeze(dim=2) for x in torch.split(latent, 1, dim=2)] # [batch, node_dim] x node
        # align_latent = list(map(lambda x, layer: layer(x), latent, model.flows))
        # causal_latent = [torch.tanh(x) for x in align_latent]
                
        with torch.autograd.set_detect_anomaly(True):    
            optimizer.zero_grad()
            
            mean, logvar, prior_logvar, latent_orig, causal_latent, align_latent, xhat = model([x_batch, y_batch])
            
            loss_ = []
            
            """reconstruction"""
            recon = 0.5 * torch.pow(xhat - x_batch, 2).sum(axis=[1, 2, 3]).mean() 
            # recon = F.mse_loss(xhat, x_batch)
            loss_.append(('recon', recon))
            
            """KL-Divergence"""
            KL = (torch.pow(mean, 2) / torch.exp(prior_logvar)).sum(axis=1)
            KL += prior_logvar.sum(axis=1)
            KL -= logvar.sum(axis=1)
            KL += torch.exp(logvar - prior_logvar).sum(axis=1)
            KL -= config["node"]
            KL *= 0.5
            KL = KL.mean()
            loss_.append(('KL', KL))
            
            """DAG reconstruction""" # prior constraint without trainable variance
            DAG_recon = 0.5 * torch.pow(latent_orig - latent_orig.matmul(model.B), 2).sum(axis=[1, 2]).mean()
            loss_.append(('DAG_recon', DAG_recon))
            
            """Label Alignment"""
            align = 0.5 * torch.pow(torch.cat(align_latent, dim=1) - y_batch, 2).sum(axis=1).mean()
            loss_.append(('align', align))
            
            loss = recon + config["beta"] * KL 
            loss += config["lambda"] * DAG_recon
            loss += config["gamma"] * align
            loss_.append(('loss', loss))
            
            loss.backward()
            optimizer.step()
        
        """accumulate losses"""
        for x, y in loss_:
            logs[x] = logs.get(x) + [y.item()]
    
    return logs, xhat
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
    class CustomDataset(Dataset): 
        def __init__(self):
            train_imgs = os.listdir('./utils/causal_data/pendulum/train')
            train_x = []
            for i in tqdm.tqdm(range(len(train_imgs)), desc="train data loading"):
                train_x.append(np.array(
                    Image.open("./utils/causal_data/pendulum/train/{}".format(train_imgs[i])).resize((96, 96))
                    )[:, :, :3])
            self.x_data = (np.array(train_x).astype(float) - 127.5) / 127.5
            
            label = np.array([x[:-4].split('_')[1:] for x in train_imgs]).astype(float)
            label = label - label.mean(axis=0)
            self.std = label.std(axis=0)
            label = label / label.std(axis=0)
            self.y_data = label.round(2)
            self.name = ['light', 'angle', 'length', 'position']

        def __len__(self): 
            return len(self.x_data)

        def __getitem__(self, idx): 
            x = torch.FloatTensor(self.x_data[idx])
            y = torch.FloatTensor(self.y_data[idx])
            return x, y
    
    dataset = CustomDataset()
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)
    
    """
    Estimated Causal Adjacency Matrix
    light -> length
    light -> position
    angle -> length
    angle -> position
    length -- position
    
    Since var(length) < var(position), we set length -> position
    """
    # dataset.std
    B = torch.zeros(config["node"], config["node"])
    B_value = 1
    B[dataset.name.index('light'), dataset.name.index('length')] = B_value
    B[dataset.name.index('light'), dataset.name.index('position')] = B_value
    B[dataset.name.index('angle'), dataset.name.index('length')] = B_value
    B[dataset.name.index('angle'), dataset.name.index('position')] = B_value
    B[dataset.name.index('length'), dataset.name.index('position')] = B_value
    # B[:2, 2:] = 1
    # B[2, 3] = 1
    
    """adjacency matrix scaling"""
    indegree = B.sum(axis=0)
    mask = (indegree != 0)
    B[:, mask] = B[:, mask] / indegree[mask]
    
    model = VAE(B, config, device)
    model.to(device)
    
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config["lr"]
    )
    
    wandb.watch(model, log_freq=100) # tracking gradients
    model.train()
    
    for epoch in range(config["epochs"]):
        logs, xhat = train(dataloader, model, config, optimizer, device)
        
        print_input = "[epoch {:03d}]".format(epoch + 1)
        print_input += ''.join([', {}: {:.4f}'.format(x, np.mean(y).round(2)) for x, y in logs.items()])
        print(print_input)
        
        if epoch % 10 == 0:
            """update log"""
            wandb.log({x : np.mean(y) for x, y in logs.items()})
            
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
    plt.savefig('./assets/image.png')
    plt.close()
    wandb.log({'reconstruction': wandb.Image(fig)})

    """model save"""
    torch.save(model.state_dict(), './assets/model_v4.pth')
    artifact = wandb.Artifact('model_v4', type='model') # description=""
    artifact.add_file('./assets/model_v4.pth')
    wandb.log_artifact(artifact)
    
    wandb.run.finish()
#%%
if __name__ == '__main__':
    main()
#%%