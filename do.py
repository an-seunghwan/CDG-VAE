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
    tags=["EDA"],
)
#%%
import argparse
def get_args(debug):
    parser = argparse.ArgumentParser('parameters')
    
    parser.add_argument('--num', type=int, default=4, 
                        help='model version')

    if debug:
        return parser.parse_args(args=[])
    else:    
        return parser.parse_args()
#%%
def main():
    #%%
    config = vars(get_args(debug=True)) # default configuration
    
    # postfix = 'activated'
    # postfix = 'mutualinfo'
    postfix = 'gam'
    
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

    """test dataset"""
    class TestCustomDataset(Dataset): 
        def __init__(self, config):
            test_imgs = [x for x in os.listdir('./utils/causal_data/pendulum/test') if x.endswith('png')]
            test_x = []
            for i in tqdm.tqdm(range(len(test_imgs)), desc="test data loading"):
                test_x.append(np.array(
                    Image.open("./utils/causal_data/pendulum/test/{}".format(test_imgs[i])).resize((config["image_size"], config["image_size"]))
                    )[:, :, :3])
            self.x_data = (np.array(test_x).astype(float) - 127.5) / 127.5
            
            label = np.array([x[:-4].split('_')[1:] for x in test_imgs]).astype(float)
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
    
    test_dataset = TestCustomDataset(config)
    test_dataloader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)
    
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
    
    """import model"""
    tmp = __import__("utils.model_{}".format(postfix), 
                    fromlist=["utils.model_{}".format(postfix)])
    VAE = getattr(tmp, "VAE")
    
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
        m = torch.zeros(config["image_size"], config["image_size"], 3)
        m[51:, ...] = 1
        mask.append(m)
        
        model = VAE(B, mask, config, device).to(device)
    else:
        model = VAE(B, config, device).to(device)
    if config["cuda"]:
        model.load_state_dict(torch.load(model_dir + '/model_{}.pth'.format(postfix)))
    else:
        model.load_state_dict(torch.load(model_dir + '/model_{}.pth'.format(postfix), map_location=torch.device('cpu')))
    
    model.eval()
    #%%
    """latent space (conditional intervention range)"""
    epsilons = []
    orig_latents = []
    latents = []
    iter_test = iter(dataloader)
    for x_batch, y_batch in tqdm.tqdm(iter_test):
        if config["cuda"]:
            x_batch = x_batch.cuda()
            y_batch = y_batch.cuda()
            
        mean, logvar, epsilon, orig_latent, latent, _ = model.encode(x_batch, deterministic=True)
        epsilons.append(epsilon.squeeze())
        orig_latents.append(orig_latent.squeeze())
        latents.append(torch.cat(latent, dim=1))
    epsilons = torch.cat(epsilons, dim=0)
    epsilons = epsilons.detach().cpu().numpy()
    orig_latents = torch.cat(orig_latents, dim=0)
    orig_latents = orig_latents.detach().cpu().numpy()
    latents = torch.cat(latents, dim=0)
    
    # orig_latents_corr = np.abs(np.corrcoef(orig_latents.T).round(2))
    # fig = plt.figure(figsize=(5, 4))
    # plt.pcolor(np.flipud(orig_latents_corr), cmap='coolwarm')
    # plt.axis('off')
    # plt.colorbar()
    # plt.show()
    # # plt.savefig('assets/epsilon_corr.png')
    # # plt.close()
    # # wandb.log({'absolute correlation plot of epsilons': wandb.Image(fig)})
    
    fig, ax = plt.subplots(2, 3, figsize=(10, 5))
    ax[0, 0].scatter(orig_latents[:, 0], orig_latents[:, 1], s=2, alpha=0.5)
    ax[0, 0].set_xlabel('$z_1$', fontsize=15)
    ax[0, 0].set_ylabel('$z_2$', fontsize=15)
    ax[0, 1].scatter(orig_latents[:, 0], orig_latents[:, 2], s=2, alpha=0.5)
    ax[0, 1].set_xlabel('$z_1$', fontsize=15)
    ax[0, 1].set_ylabel('$z_3$', fontsize=15)
    ax[0, 2].scatter(orig_latents[:, 0], orig_latents[:, 3], s=2, alpha=0.5)
    ax[0, 2].set_xlabel('$z_1$', fontsize=15)
    ax[0, 2].set_ylabel('$z_4$', fontsize=15)
    ax[1, 1].scatter(orig_latents[:, 1], orig_latents[:, 2], s=2, alpha=0.5)
    ax[1, 1].set_xlabel('$z_2$', fontsize=15)
    ax[1, 1].set_ylabel('$z_3$', fontsize=15)
    ax[1, 2].scatter(orig_latents[:, 1], orig_latents[:, 3], s=2, alpha=0.5)
    ax[1, 2].set_xlabel('$z_2$', fontsize=15)
    ax[1, 2].set_ylabel('$z_4$', fontsize=15)
    plt.tight_layout()
    plt.savefig('{}/latent_space.png'.format(model_dir), bbox_inches='tight')
    # plt.show()
    plt.close()
    
    wandb.log({'latent space (conditional intervention range)': wandb.Image(fig)})
    #%%
    """causal latent max-min difference"""
    causal_min = np.quantile(orig_latents, q=0.05, axis=0)
    causal_max = np.quantile(orig_latents, q=0.95, axis=0)
    
    transformed_causal_min = np.quantile(latents.detach().numpy(), q=0.05, axis=0)
    transformed_causal_max = np.quantile(latents.detach().numpy(), q=0.95, axis=0)
    
    fig, ax = plt.subplots(1, 2, figsize=(6, 3))
    ax[0].bar(np.arange(config["node"]), np.abs(causal_max - causal_min),
            width=0.2)
    ax[0].set_xticks(np.arange(config["node"]))
    ax[0].set_xticklabels(dataset.name)
    ax[0].set_ylabel('latent (intervened)', fontsize=12)
    ax[1].bar(np.arange(config["node"]), np.abs(transformed_causal_max - transformed_causal_min),
            width=0.2)
    ax[1].set_xticks(np.arange(config["node"]))
    ax[1].set_xticklabels(dataset.name)
    ax[1].set_ylabel('transformed latent', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('{}/latent_maxmin.png'.format(model_dir), bbox_inches='tight')
    # plt.show()
    plt.close()
    
    wandb.log({'causal latent max-min difference': wandb.Image(fig)})
    #%%
    """
    & posterior conditional variance
    & cross entropy of supervised loss
    """
    logvars = []
    align_latents = []
    iter_test = iter(dataloader)
    for x_batch, y_batch in tqdm.tqdm(iter_test):
        if config["cuda"]:
            x_batch = x_batch.cuda()
            y_batch = y_batch.cuda()
            
        mean, logvar, epsilon, orig_latent, latent, logdet, align_latent, xhat_separated, xhat = model(x_batch, deterministic=True)
        logvars.append(logvar)
        align_latents.append(torch.cat(align_latent, dim=1))
    
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
    """dependency of decoder on latent"""
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    iter_test = iter(dataloader)
    count = 1
    for _ in range(count):
        x_batch, y_batch = next(iter_test)
    if config["cuda"]:
        x_batch = x_batch.cuda()
        y_batch = y_batch.cuda()
    
    mean, logvar, epsilon, orig_latent, latent, logdet, align_latent, xhat_separated, xhat = model(x_batch, deterministic=True)
    
    latent_copy = [x.clone() for x in latent]
    latent_copy[:2] = [torch.zeros(1, 1)] * 2
    
    if postfix == 'gam':
        xhat_copy = [D(z) for D, z in zip(model.decoder, latent_copy)]
        xhat_copy = [x.view(-1, config["image_size"], config["image_size"], 3) for x in xhat_copy]
        xhat_copy = [x * m for x, m in zip(xhat_copy, model.mask)] # masking
        xhat_copy = torch.tanh(sum(xhat_copy)) # generalized addictive model (GAM)
    else:
        xhat_copy = model.decoder(torch.cat(latent_copy, dim=1)).view(1, config['image_size'], config['image_size'], 3)
    
    fig, ax = plt.subplots(2, 3, figsize=(6, 3))
    ax[0, 0].imshow((xhat[0].cpu().detach().numpy() + 1) / 2)
    # ax[0, 0].axis('off')
    ax[1, 0].plot([x[0][0].item() for x in latent])
    ax[1, 0].set_ylim(-2, 2)
    ax[0, 1].imshow((xhat_copy[0].cpu().detach().numpy() + 1) / 2)
    # ax[0, 1].axis('off')
    ax[1, 1].plot([x[0][0].item() for x in latent_copy])
    ax[1, 1].set_ylim(-2, 2)
    ax[0, 2].imshow(((xhat - xhat_copy).abs().detach().numpy()[0] + 1) / 2)
    ax[0, 2].axis('off')
    # ax[1, 2].plot([x[0][0].item() - y[0][0].item() for x, y in zip(latent, latent_copy)])
    # ax[1, 2].set_ylim(-2, 2)
    fig.delaxes(ax[1, 2])
    plt.tight_layout()
    plt.savefig('{}/latent_dependency_root.png'.format(model_dir), bbox_inches='tight')
    # plt.show()
    plt.close()
    
    wandb.log({'dependency of decoder on latent (root)': wandb.Image(fig)})
    
    if postfix == 'gam':
        xhat_copy = [D(z) for D, z in zip(model.decoder, latent_copy)]
        xhat_copy = [x.view(-1, config["image_size"], config["image_size"], 3) for x in xhat_copy]
        xhat_copy = [x * m for x, m in zip(xhat_copy, model.mask)] # masking
        xhat_copy = torch.tanh(sum(xhat_copy)) # generalized addictive model (GAM)
    else:
        xhat_copy = model.decoder(torch.cat(latent_copy, dim=1)).view(1, config['image_size'], config['image_size'], 3)
    
    fig, ax = plt.subplots(2, 3, figsize=(6, 3))
    ax[0, 0].imshow((xhat[0].cpu().detach().numpy() + 1) / 2)
    # ax[0, 0].axis('off')
    ax[1, 0].plot([x[0][0].item() for x in latent])
    ax[1, 0].set_ylim(-2, 2)
    ax[0, 1].imshow((xhat_copy[0].cpu().detach().numpy() + 1) / 2)
    # ax[0, 1].axis('off')
    ax[1, 1].plot([x[0][0].item() for x in latent_copy])
    ax[1, 1].set_ylim(-2, 2)
    ax[0, 2].imshow(((xhat - xhat_copy).abs().detach().numpy()[0] + 1) / 2)
    ax[0, 2].axis('off')
    # ax[1, 2].plot([x[0][0].item() - y[0][0].item() for x, y in zip(latent, latent_copy)])
    # ax[1, 2].set_ylim(-2, 2)
    fig.delaxes(ax[1, 2])
    plt.tight_layout()
    plt.savefig('{}/latent_dependency_child.png'.format(model_dir), bbox_inches='tight')
    # plt.show()
    plt.close()
    
    wandb.log({'dependency of decoder on latent (child)': wandb.Image(fig)})
    #%%
    """reconstruction"""
    # test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    iter_test = iter(dataloader)
    count = 2
    for _ in range(count):
        x_batch, y_batch = next(iter_test)
    if config["cuda"]:
        x_batch = x_batch.cuda()
        y_batch = y_batch.cuda()
    
    mean, logvar, epsilon, orig_latent, latent, logdet, align_latent, xhat_separated, xhat = model(x_batch, deterministic=True)
    
    # using only mean
    epsilon = mean
    # noise = torch.randn(1, config["node"] * config["node_dim"]).to(device) 
    # epsilon = mean + torch.exp(logvar / 2) * noise
    # epsilon = epsilon.view(config["node"], config["node_dim"]).contiguous()
    
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
        fig, ax = plt.subplots(2, 2, figsize=(5, 5))
        for i in range(config["node"]):
            ax.flatten()[i].imshow(xhats[i].detach().cpu().numpy())
        
        plt.tight_layout()
        plt.savefig('{}/gam.png'.format(model_dir), bbox_inches='tight')
        # plt.show()
        plt.close()
        
        wandb.log({'gam': wandb.Image(fig)})
    #%%
    """reconstruction with do-intervention"""
    for do_index, (min, max) in enumerate(zip(transformed_causal_min, transformed_causal_max)):
        fig, ax = plt.subplots(3, 3, figsize=(5, 5))
        
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
                do_xhat = [D(z_) for D, z_ in zip(model.decoder, z)]
                do_xhat = [x.view(config["image_size"], config["image_size"], 3) for x in do_xhat]
                do_xhat = [x * m for x, m in zip(do_xhat, model.mask)] # masking
                do_xhat = torch.tanh(sum(do_xhat)) # generalized addictive model (GAM)
            else:
                do_xhat = model.decoder(torch.cat(z, dim=1)).view(config["image_size"], config["image_size"], 3)
            
            ax.flatten()[k].imshow((do_xhat.clone().detach().cpu().numpy() + 1) / 2)
            ax.flatten()[k].axis('off')
            ax.flatten()[k].set_title('x = {}'.format(do_value))
            # ax.flatten()[k].set_title('do({} = {})'.format(name[do_index], do_value))
        
        plt.suptitle('do({} = x)'.format(test_dataset.name[do_index]), fontsize=15)
        plt.savefig('{}/do_{}.png'.format(model_dir, test_dataset.name[do_index]), bbox_inches='tight')
        # plt.show()
        plt.close()
        
        wandb.log({'do intervention on {}'.format(test_dataset.name[do_index]): wandb.Image(fig)})
    #%%
    wandb.run.finish()
#%%
if __name__ == '__main__':
    main()
#%%
# fig, ax = plt.subplots(2, 5, figsize=(10, 5))
# for i, k in enumerate(np.linspace(-2, 4, 10)):
#     z = [torch.tensor([[0]], dtype=torch.float32),
#         torch.tensor([[0]], dtype=torch.float32),
#         torch.tensor([[k]], dtype=torch.float32),
#         torch.tensor([[0]], dtype=torch.float32)]
#     z = list(map(lambda x, layer: layer(x), z, model.flows))
#     z = [z_[0] for z_ in z]
    
#     do_xhat = model.decoder(torch.cat(z, dim=1)).view(config["image_size"], config["image_size"], 3)
#     ax.flatten()[i].imshow((do_xhat.detach().cpu().numpy() + 1) / 2)
# plt.show()
# plt.close()

# #
# points = [[2, -1.5], [2, 3], [0, 1], [0, 3]]
# plt.scatter(epsilons[:, 0], epsilons[:, 2], s=2, alpha=0.5)
# plt.scatter([p[0] for p in points], [p[1] for p in points])
# plt.xlabel('$\epsilon_1$', fontsize=15)
# plt.ylabel('$\epsilon_3$', fontsize=15)
# plt.show()
# plt.close()

# fig, ax = plt.subplots(2, 2, figsize=(5, 5))    
# do_xhats = []
# for i, p in enumerate(points):
#     z = [torch.tensor([[0]], dtype=torch.float32),
#         torch.tensor([[0]], dtype=torch.float32),
#         torch.tensor([[0]], dtype=torch.float32),
#         torch.tensor([[0]], dtype=torch.float32)]
#     z[0] = torch.tensor([[p[0]]], dtype=torch.float32)
#     z[2] = torch.tensor([[p[1]]], dtype=torch.float32)
#     z = list(map(lambda x, layer: layer(x), z, model.flows))
#     z = [z_[0] for z_ in z]
    
#     do_xhat = model.decoder(torch.cat(z, dim=1)).view(config["image_size"], config["image_size"], 3)
#     ax.flatten()[i].imshow((do_xhat.detach().cpu().numpy() + 1) / 2)
#     do_xhats.append(do_xhat)
# plt.show()
# plt.close()

# fig, ax = plt.subplots(1, 2, figsize=(5, 5))    
# ax.flatten()[0].imshow(((do_xhats[0] - do_xhats[1]).detach().cpu().numpy() + 1) / 2)
# ax.flatten()[1].imshow(((do_xhats[2] - do_xhats[3]).detach().cpu().numpy() + 1) / 2)
# plt.show()
# plt.close()
#%%