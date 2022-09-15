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

from utils.model_mutualinfo import (
    VAE,
    Discriminator
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

wandb.init(
    project="(proposal)CausalVAE", 
    entity="anseunghwan",
    tags=["Mutual-Information"],
)
#%%
import argparse
def get_args(debug):
    parser = argparse.ArgumentParser('parameters')
    
    parser.add_argument('--seed', type=int, default=1, 
                        help='seed for repeatable results')

    parser.add_argument("--node", default=4, type=int,
                        help="the number of nodes")
    parser.add_argument("--scm", default='nonlinear', type=str,
                        help="SCM structure options: linear or nonlinear")
    parser.add_argument("--flow_num", default=1, type=int,
                        help="the number of invertible NN flow")
    parser.add_argument("--inverse_loop", default=100, type=int,
                        help="the number of inverse loop")
    
    parser.add_argument("--label_normalization", default=True, type=bool,
                        help="If True, normalize additional information label data")
    parser.add_argument("--adjacency_scaling", default=True, type=bool,
                        help="If True, scaling adjacency matrix with in-degree")
    
    parser.add_argument('--image_size', default=64, type=int,
                        help='width and heigh of image')
    
    parser.add_argument('--epochs', default=100, type=int,
                        help='maximum iteration')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='batch size')
    parser.add_argument('--lr', default=0.001, type=float,
                        help='learning rate')
    parser.add_argument('--lr_D', default=0.0001, type=float,
                        help='learning rate for discriminator')
    
    parser.add_argument('--beta', default=10, type=float,
                        help='observation noise')
    parser.add_argument('--lambda', default=10, type=float,
                        help='weight of label alignment loss')
    parser.add_argument('--gamma', default=10, type=float,
                        help='weight of f-divergence (lower bound of information)')
    
    if debug:
        return parser.parse_args(args=[])
    else:    
        return parser.parse_args()
#%%
def permute_dims(z, device):
    B, _ = z.size()
    perm = torch.randperm(B).to(device)
    perm_z = z[perm]
    return perm_z

def train(dataloader, model, discriminator, config, optimizer, optimizer_D, device):
    logs = {
        'loss': [], 
        'recon': [],
        'KL': [],
        'alignment': [],
        'MutualInfo': [],
    }
    # for debugging
    for i in range(config["node"]):
        logs['posterior_variance{}'.format(i+1)] = []
    
    for (x_batch, y_batch) in tqdm.tqdm(iter(dataloader), desc="inner loop"):
        
        if config["cuda"]:
            x_batch = x_batch.cuda()
            y_batch = y_batch.cuda()
        
        with torch.autograd.set_detect_anomaly(True):    
            mean, logvar, epsilon, _, _, _, align_latent, xhat = model(x_batch)
            
            loss_ = []
            
            """reconstruction"""
            recon = 0.5 * torch.pow(xhat - x_batch, 2).sum(axis=[1, 2, 3]).mean() 
            # recon = F.mse_loss(xhat, x_batch)
            loss_.append(('recon', recon))
            
            """KL-Divergence"""
            KL = torch.pow(mean, 2).sum(axis=1)
            KL -= logvar.sum(axis=1)
            KL += torch.exp(logvar).sum(axis=1)
            KL -= config["node"]
            KL *= 0.5
            KL = KL.mean()
            loss_.append(('KL', KL))
            
            """Label Alignment : CrossEntropy"""
            y_hat = torch.sigmoid(torch.cat(align_latent, dim=1))
            align = F.binary_cross_entropy(y_hat, y_batch, reduction='none').sum(axis=1).mean()
            loss_.append(('alignment', align))
            
            """mutual information"""
            D_joint = discriminator(x_batch, epsilon)
            epsilon_perm = permute_dims(epsilon, device)
            D_marginal = discriminator(x_batch, epsilon_perm)
            MI = -(D_joint.mean() - torch.exp(D_marginal - 1).mean())
            loss_.append(('MutualInfo', MI))
            
            ### posterior variance: for debugging
            logvar_ = logvar.mean(axis=0)
            for i in range(config["node"]):
                loss_.append(('posterior_variance{}'.format(i+1), torch.exp(logvar_[i])))
            
            loss = recon + config["beta"] * KL 
            loss += config["lambda"] * align
            loss += config["gamma"] * MI
            loss_.append(('loss', loss))
            
            optimizer.zero_grad()
            optimizer_D.zero_grad()
            loss.backward(retain_graph=True)
            MI.backward()
            optimizer.step()
            optimizer_D.step()
            
        """accumulate losses"""
        for x, y in loss_:
            logs[x] = logs.get(x) + [y.item()]
    
    return logs, xhat
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

    """dataset"""
    class CustomDataset(Dataset): 
        def __init__(self, config):
            train_imgs = [x for x in os.listdir('./utils/causal_data/pendulum/train') if x.endswith('png')]
            train_x = []
            for i in tqdm.tqdm(range(len(train_imgs)), desc="train data loading"):
                train_x.append(np.array(
                    Image.open("./utils/causal_data/pendulum/train/{}".format(train_imgs[i])).resize((config["image_size"], config["image_size"]))
                    )[:, :, :3])
            self.x_data = (np.array(train_x).astype(float) - 127.5) / 127.5
            
            label = np.array([x[:-4].split('_')[1:] for x in train_imgs]).astype(float)
            label = label - label.mean(axis=0)
            self.std = label.std(axis=0)
            """bounded label: normalize to (0, 1)"""
            if config["label_normalization"]: 
                label = (label - label.min(axis=0)) / (label.max(axis=0) - label.min(axis=0))
            self.y_data = label
            self.name = ['light', 'angle', 'length', 'position']

        def __len__(self): 
            return len(self.x_data)

        def __getitem__(self, idx): 
            x = torch.FloatTensor(self.x_data[idx])
            y = torch.FloatTensor(self.y_data[idx])
            return x, y
    
    dataset = CustomDataset(config)
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)
    
    """
    Estimated Causal Adjacency Matrix
    light -> length
    light -> position
    angle -> length
    angle -> position
    length -- position
    
    # Since var(length) < var(position), we set length -> position
    """
    # dataset.std
    B = torch.zeros(config["node"], config["node"])
    B_value = 1
    B[dataset.name.index('light'), dataset.name.index('length')] = B_value
    B[dataset.name.index('light'), dataset.name.index('position')] = B_value
    B[dataset.name.index('angle'), dataset.name.index('length')] = B_value
    B[dataset.name.index('angle'), dataset.name.index('position')] = B_value
    # B[dataset.name.index('length'), dataset.name.index('position')] = B_value
    
    """adjacency matrix scaling"""
    if config["adjacency_scaling"]:
        indegree = B.sum(axis=0)
        mask = (indegree != 0)
        B[:, mask] = B[:, mask] / indegree[mask]
    
    model = VAE(B, config, device) 
    model = model.to(device)
    discriminator = Discriminator(config, device)
    discriminator = discriminator.to(device)
    
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config["lr"]
    )
    optimizer_D = torch.optim.Adam(
        discriminator.parameters(), 
        lr=config["lr_D"]
    )
    
    wandb.watch(model, log_freq=100) # tracking gradients
    wandb.watch(discriminator, log_freq=100) # tracking gradients
    print(model.train())
    print(discriminator.train())
    
    for epoch in range(config["epochs"]):
        logs, xhat = train(dataloader, model, discriminator, config, optimizer, optimizer_D, device)
        
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
    plt.savefig('./assets/recon.png')
    plt.close()
    wandb.log({'reconstruction': wandb.Image(fig)})
    
    """model save"""
    torch.save(model.state_dict(), './assets/model_{}.pth'.format('mutualinfo'))
    artifact = wandb.Artifact('model_{}'.format('mutualinfo'), 
                              type='model',
                              metadata=config) # description=""
    artifact.add_file('./assets/model_{}.pth'.format('mutualinfo'))
    artifact.add_file('./main_{}.py'.format('mutualinfo'))
    artifact.add_file('./utils/model_{}.py'.format('mutualinfo'))
    wandb.log_artifact(artifact)
    
    wandb.run.finish()
#%%
if __name__ == '__main__':
    main()
#%%