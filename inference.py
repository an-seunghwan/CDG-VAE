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
    LabeledDataset, 
    UnLabeledDataset,
)
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
    project="CausalDisentangled", 
    entity="anseunghwan",
    tags=["VAEBased", "Inference"],
)
#%%
import argparse
def get_args(debug):
    parser = argparse.ArgumentParser('parameters')
    
    parser.add_argument('--num', type=int, default=0, 
                        help='model version')

    if debug:
        return parser.parse_args(args=[])
    else:    
        return parser.parse_args()
#%%
def main():
    #%%
    config = vars(get_args(debug=False)) # default configuration
    
    # model_name = 'VAE'
    # model_name = 'InfoMax'
    # model_name = 'CDGVAE'
    model_name = 'CDGVAEsemi'
    
    # scm = 'linear'
    scm = 'nonlinear'
    
    """model load"""
    artifact = wandb.use_artifact('anseunghwan/CausalDisentangled/model_{}_{}:v{}'.format(model_name, scm, config["num"]), type='model')
    for key, item in artifact.metadata.items():
        config[key] = item
    assert model_name == config["model"]
    assert scm == config["scm"]
    model_dir = artifact.download()
    
    config["cuda"] = torch.cuda.is_available()
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    wandb.config.update(config)
    
    set_random_seed(config["seed"])
    torch.manual_seed(config["seed"])
    if config["cuda"]:
        torch.cuda.manual_seed(config["seed"])

    """dataset"""
    dataset = LabeledDataset(config)
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False)

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
    
    """import model"""
    if config["model"] == 'VAE':
        from modules.model import VAE
        model = VAE(B, config, device) 
        
    elif config["model"] == 'InfoMax':
        from modules.model import VAE
        model = VAE(B, config, device) 
        
    elif config["model"] in ['CDGVAE', 'CDGVAEsemi']:
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
        
        from modules.model import CDGVAE
        model = CDGVAE(B, mask, config, device) 
    
    else:
        raise ValueError('Not supported model!')
        
    model = model.to(device)
    
    if config["cuda"]:
        model.load_state_dict(torch.load(model_dir + '/model_{}_{}.pth'.format(config["model"], config["scm"])))
    else:
        model.load_state_dict(torch.load(model_dir + '/model_{}_{}.pth'.format(config["model"], config["scm"]), 
                                         map_location=torch.device('cpu')))
    
    model.eval()
    #%%
    """
    & intervention range
    & posterior conditional variance
    & cross entropy of supervised loss: disentanglement
    """
    epsilons = []
    orig_latents = []
    latents = []
    logvars = []
    align_latents = []
    for x_batch, y_batch in tqdm.tqdm(iter(dataloader)):
        if config["cuda"]:
            x_batch = x_batch.cuda()
            y_batch = y_batch.cuda()
        
        with torch.no_grad():
            if config["model"] in ['CDGVAE', 'CDGVAEsemi']:
                mean, logvar, epsilon, orig_latent, latent, logdet, align_latent, xhat_separated, xhat = model(x_batch, deterministic=True)
            else:
                mean, logvar, epsilon, orig_latent, latent, logdet, align_latent, xhat = model(x_batch, deterministic=True)
        epsilons.append(epsilon.squeeze())
        orig_latents.append(orig_latent.squeeze())
        latents.append(torch.cat(latent, dim=1))
        logvars.append(logvar)
        align_latents.append(torch.cat(align_latent, dim=1))
    
    epsilons = torch.cat(epsilons, dim=0)
    epsilons = epsilons.cpu().numpy()
    orig_latents = torch.cat(orig_latents, dim=0)
    orig_latents = orig_latents.cpu().numpy()
    latents = torch.cat(latents, dim=0)
    
    ### intervention range
    causal_min = np.min(orig_latents, axis=0)
    causal_max = np.max(orig_latents, axis=0)
    transformed_causal_min = np.min(latents.cpu().numpy(), axis=0)
    transformed_causal_max = np.max(latents.cpu().numpy(), axis=0)
    # causal_min = np.quantile(orig_latents, q=0.01, axis=0)
    # causal_max = np.quantile(orig_latents, q=0.99, axis=0)
    # transformed_causal_min = np.quantile(latents.cpu().numpy(), q=0.01, axis=0)
    # transformed_causal_max = np.quantile(latents.cpu().numpy(), q=0.99, axis=0)
    
    fig, ax = plt.subplots(1, 2, figsize=(6, 3))
    diff = np.abs(causal_max - causal_min)
    # diff /= diff.max()
    ax[0].bar(np.arange(config["node"]), diff, width=0.2)
    ax[0].set_xticks(np.arange(config["node"]))
    ax[0].set_xticklabels(dataset.name[:config["node"]])
    ax[0].set_ylabel('latent (intervened)', fontsize=12)
    diff = np.abs(transformed_causal_max - transformed_causal_min)
    # diff /= diff.max()
    ax[1].bar(np.arange(config["node"]), diff, width=0.2)
    ax[1].set_xticks(np.arange(config["node"]))
    ax[1].set_xticklabels(dataset.name[:config["node"]])
    ax[1].set_ylabel('transformed latent', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('{}/latent_maxmin.png'.format(model_dir), bbox_inches='tight')
    # plt.show()
    plt.close()
    
    wandb.log({'causal latent max-min difference': wandb.Image(fig)})
    
    ### posterior conditional variance
    logvars = torch.cat(logvars, dim=0)
    fig = plt.figure(figsize=(5, 3))
    plt.bar(np.arange(config["node"]), torch.exp(logvars).mean(axis=0).cpu().numpy(),
            width=0.2)
    plt.xticks(np.arange(config["node"]), dataset.name[:config["node"]])
    # plt.xlabel('node', fontsize=12)
    plt.ylabel('posterior variance', fontsize=12)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig('{}/posterior_variance.png'.format(model_dir), bbox_inches='tight')
    # plt.show()
    plt.close()
    
    wandb.log({'posterior conditional variance': wandb.Image(fig)})
    
    ### cross entropy
    align_latents = torch.cat(align_latents, dim=0)
    y_hat = torch.sigmoid(align_latents)
    align = F.binary_cross_entropy(y_hat.cpu(), 
                                   torch.tensor(dataset.y_data[:, :4], dtype=torch.float32), 
                                   reduction='none').mean(axis=0)
    fig = plt.figure(figsize=(5, 3))
    plt.bar(np.arange(config["node"]), align.cpu().numpy(),
            width=0.2)
    plt.xticks(np.arange(config["node"]), dataset.name[:config["node"]])
    # plt.xlabel('node', fontsize=12)
    plt.ylabel('latent', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('{}/crossentropy.png'.format(model_dir), bbox_inches='tight')
    # plt.show()
    plt.close()
    
    wandb.log({'cross entropy of supervised loss': wandb.Image(fig)})
    #%%
    """reconstruction"""
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    iter_test = iter(dataloader)
    count = 8
    for _ in range(count):
        x_batch, y_batch = next(iter_test)
    if config["cuda"]:
        x_batch = x_batch.cuda()
        y_batch = y_batch.cuda()
    
    with torch.no_grad():
        if config["model"] in ['CDGVAE', 'CDGVAEsemi']:
            mean, logvar, epsilon, orig_latent, latent, logdet, align_latent, xhat_separated, xhat = model(x_batch, deterministic=True)
        else:
            mean, logvar, epsilon, orig_latent, latent, logdet, align_latent, xhat = model(x_batch, deterministic=True)
    
    epsilon = mean # using only mean
    
    fig, ax = plt.subplots(1, 2, figsize=(4, 4))
    
    ax[0].imshow((x_batch[0].cpu().numpy() + 1) / 2)
    ax[0].axis('off')
    ax[0].set_title('original')
    
    ax[1].imshow((xhat[0].cpu().numpy() + 1) / 2)
    ax[1].axis('off')
    ax[1].set_title('recon')
    
    plt.tight_layout()
    plt.savefig('{}/original_and_recon.png'.format(model_dir), bbox_inches='tight')
    # plt.show()
    plt.close()
    
    wandb.log({'original and reconstruction': wandb.Image(fig)})
    
    if config["model"] in ['CDGVAE', 'CDGVAEsemi']:
        xhats = [x.view(model.config["image_size"], model.config["image_size"], 3) for x in xhat_separated]
        fig, ax = plt.subplots(1, 3, figsize=(7, 4))
        for i in range(len(config["factor"])):
            ax[i].imshow(xhats[i].cpu().numpy())
        
        plt.tight_layout()
        plt.savefig('{}/gam.png'.format(model_dir), bbox_inches='tight')
        # plt.show()
        plt.close()
        
        wandb.log({'gam': wandb.Image(fig)})
    #%%
    """do-intervention"""
    fig, ax = plt.subplots(config["node"], 7, figsize=(7, 4))
    
    with torch.no_grad():
        for do_index, (min, max) in enumerate(zip(transformed_causal_min, transformed_causal_max)):
            for k, do_value in enumerate(np.linspace(min, max, 7)):
                do_value = round(do_value, 1)
                latent_ = [x.clone() for x in latent]
                latent_[do_index] = torch.tensor([[do_value]], dtype=torch.float32).to(device)
                z = model.inverse(latent_)
                z = torch.cat(z, dim=1).clone()
                for j in range(config["node"]):
                    if j == do_index:
                        continue
                    else:
                        if j == 0:  # root node
                            z[:, j] = epsilon[:, j]
                        z[:, j] = torch.matmul(z[:, :j], model.B[:j, j]) + epsilon[:, j]
                z = torch.split(z, 1, dim=1)
                z = list(map(lambda x, layer: layer(x), z, model.flows))
                z = [z_[0] for z_ in z]
                
                if config["model"] in ['CDGVAE', 'CDGVAEsemi']:
                    _, do_xhat = model.decode(z)
                    do_xhat = do_xhat[0]
                else:
                    do_xhat = model.decoder(torch.cat(z, dim=1)).view(config["image_size"], config["image_size"], 3)
                
                ax[do_index, k].imshow((do_xhat.clone().cpu().numpy() + 1) / 2)
                ax[do_index, k].axis('off')
    
    plt.tight_layout()
    plt.savefig('{}/do.png'.format(model_dir), bbox_inches='tight')
    # plt.show()
    plt.close()
    
    wandb.log({'do intervention ({})'.format(', '.join(dataset.name[:config["node"]])): wandb.Image(fig)})
    #%%
    wandb.run.finish()
#%%
if __name__ == '__main__':
    main()
#%%