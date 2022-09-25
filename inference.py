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

from utils.simulation import (
    set_random_seed,
    is_dag,
)

from utils.viz import (
    viz_graph,
    viz_heatmap,
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
    project="(proposal)CausalVAE", 
    entity="anseunghwan",
    tags=["Inference"],
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
    
    postfix = 'vanilla' # 18
    # postfix = 'InfoMax' # 5
    # postfix = 'gam' # 16
    
    """model load"""
    artifact = wandb.use_artifact('anseunghwan/(proposal)CausalVAE/model_{}:v{}'.format(postfix, config["num"]), type='model')
    for key, item in artifact.metadata.items():
        config[key] = item
    model_dir = artifact.download()
    
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
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False)

    # """test dataset"""
    # class TestCustomDataset(Dataset): 
    #     def __init__(self, config):
    #         test_imgs = [x for x in os.listdir('./utils/causal_data/pendulum/test') if x.endswith('png')]
    #         test_x = []
    #         for i in tqdm.tqdm(range(len(test_imgs)), desc="test data loading"):
    #             test_x.append(np.array(
    #                 Image.open("./utils/causal_data/pendulum/test/{}".format(test_imgs[i])).resize((config["image_size"], config["image_size"]))
    #                 )[:, :, :3])
    #         self.x_data = (np.array(test_x).astype(float) - 127.5) / 127.5
            
    #         label = np.array([x[:-4].split('_')[1:] for x in test_imgs]).astype(float)
    #         label = label - label.mean(axis=0)
    #         self.std = label.std(axis=0)
    #         """bounded label: normalize to (0, 1)"""
    #         if config["label_normalization"]: 
    #             label = (label - label.min(axis=0)) / (label.max(axis=0) - label.min(axis=0))
    #         self.y_data = label
    #         self.name = ['light', 'angle', 'length', 'position']

    #     def __len__(self): 
    #         return len(self.x_data)

    #     def __getitem__(self, idx): 
    #         x = torch.FloatTensor(self.x_data[idx])
    #         y = torch.FloatTensor(self.y_data[idx])
    #         return x, y
    
    # test_dataset = TestCustomDataset(config)
    # test_dataloader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)
    
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
    if postfix == 'gam':
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
        
        tmp = __import__("utils.model_{}".format(postfix), 
                        fromlist=["utils.model_{}".format(postfix)])
        VAE = getattr(tmp, "VAE")
        model = VAE(B, mask, config, device).to(device)
    else:
        from utils.model_base import VAE
        model = VAE(B, config, device).to(device)
    
    if config["cuda"]:
        model.load_state_dict(torch.load(model_dir + '/model_{}.pth'.format(postfix)))
    else:
        model.load_state_dict(torch.load(model_dir + '/model_{}.pth'.format(postfix), map_location=torch.device('cpu')))
    
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
        
        if postfix == 'gam':    
            mean, logvar, epsilon, orig_latent, latent, logdet, align_latent, xhat_separated, xhat = model(x_batch, deterministic=True)
        else:
            mean, logvar, epsilon, orig_latent, latent, logdet, align_latent, xhat = model(x_batch, deterministic=True)
        epsilons.append(epsilon.squeeze())
        orig_latents.append(orig_latent.squeeze())
        latents.append(torch.cat(latent, dim=1))
        logvars.append(logvar)
        align_latents.append(torch.cat(align_latent, dim=1))
    
    epsilons = torch.cat(epsilons, dim=0)
    epsilons = epsilons.detach().cpu().numpy()
    orig_latents = torch.cat(orig_latents, dim=0)
    orig_latents = orig_latents.detach().cpu().numpy()
    latents = torch.cat(latents, dim=0)
    
    ### intervention range
    causal_min = np.min(orig_latents, axis=0)
    causal_max = np.max(orig_latents, axis=0)
    transformed_causal_min = np.min(latents.detach().numpy(), axis=0)
    transformed_causal_max = np.max(latents.detach().numpy(), axis=0)
    # causal_min = np.quantile(orig_latents, q=0.01, axis=0)
    # causal_max = np.quantile(orig_latents, q=0.99, axis=0)
    # transformed_causal_min = np.quantile(latents.detach().numpy(), q=0.01, axis=0)
    # transformed_causal_max = np.quantile(latents.detach().numpy(), q=0.99, axis=0)
    
    fig, ax = plt.subplots(1, 2, figsize=(6, 3))
    diff = np.abs(causal_max - causal_min)
    # diff /= diff.max()
    ax[0].bar(np.arange(config["node"]), diff, width=0.2)
    ax[0].set_xticks(np.arange(config["node"]))
    ax[0].set_xticklabels(dataset.name)
    ax[0].set_ylabel('latent (intervened)', fontsize=12)
    diff = np.abs(transformed_causal_max - transformed_causal_min)
    # diff /= diff.max()
    ax[1].bar(np.arange(config["node"]), diff, width=0.2)
    ax[1].set_xticks(np.arange(config["node"]))
    ax[1].set_xticklabels(dataset.name)
    ax[1].set_ylabel('transformed latent', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('{}/latent_maxmin.png'.format(model_dir), bbox_inches='tight')
    # plt.show()
    plt.close()
    
    wandb.log({'causal latent max-min difference': wandb.Image(fig)})
    
    ### posterior conditional variance
    logvars = torch.cat(logvars, dim=0)
    fig = plt.figure(figsize=(5, 3))
    plt.bar(np.arange(config["node"]), torch.exp(logvars).mean(axis=0).detach().numpy(),
            width=0.2)
    plt.xticks(np.arange(config["node"]), dataset.name)
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
    align = F.binary_cross_entropy(y_hat, 
                                   torch.tensor(dataset.y_data, dtype=torch.float32), 
                                   reduction='none').mean(axis=0)
    fig = plt.figure(figsize=(5, 3))
    plt.bar(np.arange(config["node"]), align.detach().numpy(),
            width=0.2)
    plt.xticks(np.arange(config["node"]), dataset.name)
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
    
    if postfix == 'gam':    
        mean, logvar, epsilon, orig_latent, latent, logdet, align_latent, xhat_separated, xhat = model(x_batch, deterministic=True)
    else:
        mean, logvar, epsilon, orig_latent, latent, logdet, align_latent, xhat = model(x_batch, deterministic=True)
    
    epsilon = mean # using only mean
    
    fig, ax = plt.subplots(1, 2, figsize=(4, 4))
    
    ax[0].imshow((x_batch[0].cpu().detach().numpy() + 1) / 2)
    ax[0].axis('off')
    ax[0].set_title('original')
    
    ax[1].imshow((xhat[0].cpu().detach().numpy() + 1) / 2)
    ax[1].axis('off')
    ax[1].set_title('recon')
    
    plt.tight_layout()
    plt.savefig('{}/original_and_recon.png'.format(model_dir), bbox_inches='tight')
    # plt.show()
    plt.close()
    
    wandb.log({'original and reconstruction': wandb.Image(fig)})
    
    if postfix == 'gam':
        xhats = [x.view(model.config["image_size"], model.config["image_size"], 3) for x in xhat_separated]
        fig, ax = plt.subplots(1, 3, figsize=(7, 4))
        for i in range(len(config["factor"])):
            ax[i].imshow(xhats[i].detach().cpu().numpy())
        
        plt.tight_layout()
        plt.savefig('{}/gam.png'.format(model_dir), bbox_inches='tight')
        # plt.show()
        plt.close()
        
        wandb.log({'gam': wandb.Image(fig)})
    #%%
    """do-intervention"""
    fig, ax = plt.subplots(config["node"], 9, figsize=(10, 4))
    
    for do_index, (min, max) in enumerate(zip(transformed_causal_min, transformed_causal_max)):
        for k, do_value in enumerate(np.linspace(min, max, 9)):
            do_value = round(do_value, 1)
            latent_ = [x.clone() for x in latent]
            latent_[do_index] = torch.tensor([[do_value]], dtype=torch.float32)
            z = model.inverse(latent_)
            z = torch.cat(z, dim=1).clone().detach()
            for j in range(config["node"]):
                if j == do_index:
                    continue
                else:
                    if j == 0:  # root node
                        z[:, j] = epsilon[:, j]
                    z[:, j] = torch.matmul(z[:, :j], B[:j, j]) + epsilon[:, j]
            z = torch.split(z, 1, dim=1)
            z = list(map(lambda x, layer: layer(x), z, model.flows))
            z = [z_[0] for z_ in z]
            
            if postfix == 'gam':
                _, do_xhat = model.decode(z)
                do_xhat = do_xhat[0]
            else:
                do_xhat = model.decoder(torch.cat(z, dim=1)).view(config["image_size"], config["image_size"], 3)
            
            ax[do_index, k].imshow((do_xhat.clone().detach().cpu().numpy() + 1) / 2)
            ax[do_index, k].axis('off')
    
    plt.savefig('{}/do.png'.format(model_dir), bbox_inches='tight')
    # plt.show()
    plt.close()
    
    wandb.log({'do intervention ({})'.format(', '.join(dataset.name)): wandb.Image(fig)})
    #%%
    wandb.run.finish()
#%%
if __name__ == '__main__':
    main()
#%%