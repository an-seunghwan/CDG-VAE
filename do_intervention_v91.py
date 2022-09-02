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

from utils.model_v91 import (
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
    tags=["GeneralizedLinearSEM", "Identifiable", "do-intervention"],
)
#%%
import argparse
def get_args(debug):
    parser = argparse.ArgumentParser('parameters')
    
    parser.add_argument('--version', type=int, default=8, 
                        help='model version')

    if debug:
        return parser.parse_args(args=[])
    else:    
        return parser.parse_args()
#%%
def main():
    config = vars(get_args(debug=True)) # default configuration
    
    """model load"""
    artifact = wandb.use_artifact('anseunghwan/(causal)VAE/model_v91:v{}'.format(config["version"]), type='model')
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
            if config["label_normalization"]:
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
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    """test dataset"""
    class TestCustomDataset(Dataset): 
        def __init__(self):
            test_imgs = list(sorted(os.listdir('./utils/causal_data/pendulum/test')))
            test_x = []
            for i in tqdm.tqdm(range(len(test_imgs)), desc="test data loading"):
                test_x.append(np.array(
                    Image.open("./utils/causal_data/pendulum/test/{}".format(test_imgs[i])).resize((96, 96))
                    )[:, :, :3])
            self.x_data = (np.array(test_x).astype(float) - 127.5) / 127.5
            
            label = np.array([x[:-4].split('_')[1:] for x in test_imgs]).astype(float)
            label = label - label.mean(axis=0)
            self.std = label.std(axis=0)
            if config["label_normalization"]:
                label = label / label.std(axis=0)
            self.y_data = label.round(2)
            self.name = ['light', 'angle', 'length', 'position']

        def __len__(self): 
            return len(self.x_data)

        def __getitem__(self, idx): 
            x = torch.FloatTensor(self.x_data[idx])
            y = torch.FloatTensor(self.y_data[idx])
            return x, y
    
    test_dataset = TestCustomDataset()
    test_dataloader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)
    
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
    
    """adjacency matrix scaling"""
    if config["adjacency_scaling"]:
        indegree = B.sum(axis=0)
        mask = (indegree != 0)
        B[:, mask] = B[:, mask] / indegree[mask]
    
    model = VAE(B, config, device).to(device)
    if config["cuda"]:
        model.load_state_dict(torch.load(model_dir + '/model_v91.pth'))
    else:
        model.load_state_dict(torch.load(model_dir + '/model_v91.pth', map_location=torch.device('cpu')))
    
    model.eval()
    
    """get intervention range"""
    causal_latents = []
    iter_test = iter(test_dataloader)
    for x_batch, y_batch in tqdm.tqdm(iter_test):
        if config["cuda"]:
            x_batch = x_batch.cuda()
            y_batch = y_batch.cuda()
            
        # mean, logvar, latent_orig, causal_latent = model.encode(x_batch, withnoise=False)
        _, _, _, _, _, _, causal_latent, _ = model([x_batch, y_batch], running=False)
        causal_latents.append(torch.cat(causal_latent, dim=1))
        
    causal_latents = torch.cat(causal_latents, dim=0)
    causal_min = np.quantile(causal_latents.detach().numpy(), q=0.1, axis=0)
    causal_max = np.quantile(causal_latents.detach().numpy(), q=0.9, axis=0)
    # causal_min = torch.min(causal_latents, axis=0).values.detach().numpy()
    # causal_max = torch.max(causal_latents, axis=0).values.detach().numpy()
    
    [x.data for x in model.running_mean]
    [x.data for x in model.running_std]
    
    """causal latent max-min difference"""
    fig = plt.figure(figsize=(5, 3))
    plt.bar(np.arange(config["node"]), np.abs(causal_max - causal_min),
            width=0.2)
    plt.xticks(np.arange(config["node"]), dataset.name)
    # plt.xlabel('node', fontsize=12)
    plt.ylabel('latent', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('{}/latent_maxmin.png'.format(model_dir), bbox_inches='tight')
    # plt.show()
    plt.close()
    
    wandb.log({'causal latent max-min difference': wandb.Image(fig)})
    
    """check exogenous"""
    unique_root = np.unique(test_dataset.y_data[:, 0])
    unique_root_idx = np.where(test_dataset.y_data[:, 0] == unique_root[0])[0]
    x_samples = torch.tensor(test_dataset.x_data[unique_root_idx, ...], dtype=torch.float32)
    y_samples = torch.tensor(test_dataset.y_data[unique_root_idx, ...], dtype=torch.float32)
    mean, logvar, prior_logvar, orig_latent, flow_latent, align_latent, causal_latent, xhat = model([x_samples, y_samples], running=False)
    
    plt.plot(mean.detach().numpy()[:, 0])
    
    torch.pow(mean, 2)
    torch.exp(logvar)
    torch.exp(prior_logvar)
    
    # prior_logvar = torch.zeros(mean.shape)
    
    KL = (torch.pow(mean, 2) / torch.exp(prior_logvar)).sum(axis=1)
    print(KL)
    KL += prior_logvar.sum(axis=1)
    print(KL)
    KL -= logvar.sum(axis=1)
    print(KL)
    KL += torch.exp(logvar - prior_logvar).sum(axis=1)
    print(KL)
    KL -= config["node"]
    print(KL)
    KL *= 0.5
    KL = KL.mean()
    print(KL)
    
    """reconstruction"""
    test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=False)
    iter_test = iter(test_dataloader)
    count = 2
    for _ in range(count):
        x_batch, y_batch = next(iter_test)
    if config["cuda"]:
        x_batch = x_batch.cuda()
        y_batch = y_batch.cuda()
    
    # mean, logvar, latent_orig, causal_latent = model.encode(x_batch, withnoise=False)
    # xhat = model.decode(causal_latent)
    mean, _, _, _, _, _, causal_latent, xhat = model([x_batch, y_batch], running=False)
    
    # using only mean
    epsilon = mean
    # noise = torch.randn(1, config["node"] * config["node_dim"]).to(device) 
    # epsilon = mean + torch.exp(logvar / 2) * noise
    # epsilon = epsilon.view(config["node"], config["node_dim"]).contiguous()
    
    fig, ax = plt.subplots(2, 2, figsize=(4, 4))
    
    ax[0, 0].imshow((x_batch[0].cpu().detach().numpy() + 1) / 2)
    ax[0, 0].axis('off')
    ax[0, 0].set_title('original')
    
    ax[0, 1].imshow((xhat[0].cpu().detach().numpy() + 1) / 2)
    ax[0, 1].axis('off')
    ax[0, 1].set_title('recon')
    
    ax[1, 0].imshow((x_batch[1].cpu().detach().numpy() + 1) / 2)
    ax[1, 0].axis('off')
    ax[1, 0].set_title('original')
    
    ax[1, 1].imshow((xhat[1].cpu().detach().numpy() + 1) / 2)
    ax[1, 1].axis('off')
    ax[1, 1].set_title('recon')
    
    plt.tight_layout()
    plt.savefig('{}/original_and_recon.png'.format(model_dir), bbox_inches='tight')
    plt.show()
    plt.close()
    
    wandb.log({'original and reconstruction': wandb.Image(fig)})
    
    """reconstruction with do-intervention"""
    for do_index, (min, max) in enumerate(zip(causal_min, causal_max)):
        fig, ax = plt.subplots(3, 3, figsize=(5, 5))
        
        # do_index = 2
        # do_value = 0.5
        for k, do_value in enumerate(np.linspace(min, max, 9)):
            do_value = round(do_value, 1)
            causal_latent_ = [x.clone() for x in causal_latent]
            causal_latent_[do_index] = torch.tensor([[do_value]], dtype=torch.float32)
            z = model.inverse(causal_latent_)
            z = torch.cat(z, dim=1).clone().detach()
            for j in range(config["node"]):
                if j == do_index:
                    continue
                else:
                    if j == 0:  # root node
                        z[:, j] = epsilon[:, j]
                    z[:, j] = torch.matmul(z[:, :j], model.B[:j, j]) + epsilon[:, j]
            z = torch.split(z, 1, dim=1)
            z = list(map(lambda x, layer: layer(x), z, model.flows))
            z = [(x - m) / s for x, m, s in zip(z, model.running_mean, model.running_std)]
            z = torch.cat([torch.tanh(x) for x in z], dim=1)
            
            do_xhat = model.decoder(z).view(96, 96, 3)

            ax.flatten()[k].imshow((do_xhat.clone().detach().cpu().numpy() + 1) / 2)
            ax.flatten()[k].axis('off')
            ax.flatten()[k].set_title('x = {}'.format(do_value))
            # ax.flatten()[k].set_title('do({} = {})'.format(name[do_index], do_value))
        
        plt.suptitle('do({} = x)'.format(test_dataset.name[do_index]), fontsize=15)
        plt.savefig('{}/do_{}.png'.format(model_dir, test_dataset.name[do_index]), bbox_inches='tight')
        # plt.show()
        plt.close()
        
        wandb.log({'do intervention on {}'.format(test_dataset.name[do_index]): wandb.Image(fig)})
    
    """for specific example"""
    iter_test = iter(test_dataloader)
    count = 25
    for _ in tqdm.tqdm(range(count)):
        x_batch, y_batch = next(iter_test)
    
    if config["cuda"]:
        x_batch = x_batch.cuda()
        y_batch = y_batch.cuda()
    
    # mean, logvar, latent_orig, causal_latent = model.encode(x_batch, withnoise=False)
    # xhat = model.decode(causal_latent)
    mean, _, _, _, _, _, causal_latent, xhat = model([x_batch, y_batch], running=False)
    
    # using only mean
    epsilon = mean
    
    fig, ax = plt.subplots(1, 6, figsize=(12, 2))
    
    ax[0].imshow((x_batch[0].cpu().detach().numpy() + 1) / 2)
    ax[0].axis('off')
    ax[0].set_title('original')
    
    ax[1].imshow((xhat[0].cpu().detach().numpy() + 1) / 2)
    ax[1].axis('off')
    ax[1].set_title('recon')
    
    # plt.tight_layout()
    # plt.show()
    # plt.close()
    
    # do-intervention
    for k, (do_index, do_value) in enumerate(zip(range(config["node"]), causal_max)):
        do_value = round(do_value, 1)
        causal_latent_ = [x.clone() for x in causal_latent]
        causal_latent_[do_index] = torch.tensor([[do_value]], dtype=torch.float32)
        z = model.inverse(causal_latent_)
        z = torch.cat(z, dim=1).clone().detach()
        for j in range(config["node"]):
            if j == do_index:
                continue
            else:
                if j == 0:  # root node
                    z[:, j] = epsilon[:, j]
                z[:, j] = torch.matmul(z[:, :j], model.B[:j, j]) + epsilon[:, j]
        z = torch.split(z, 1, dim=1)
        z = list(map(lambda x, layer: layer(x), z, model.flows))
        z = [(x - m) / s for x, m, s in zip(z, model.running_mean, model.running_std)]
        z = torch.cat([torch.tanh(x) for x in z], dim=1)
        
        do_xhat = model.decoder(z).view(96, 96, 3)

        ax.flatten()[k+2].imshow((do_xhat.clone().detach().cpu().numpy() + 1) / 2)
        ax.flatten()[k+2].axis('off')
        ax.flatten()[k+2].set_title('do({} = {:.1f})'.format(test_dataset.name[do_index], do_value))
    
    # plt.suptitle('do({} = x)'.format(name[do_index]), fontsize=15)
    plt.savefig('{}/intervention_result.png'.format(model_dir), bbox_inches='tight')
    # plt.show()
    plt.close()
    
    wandb.log({'do intervention': wandb.Image(fig)})
        
    # """for several examples"""
    # if not os.path.exists('./assets/do_intervention'): 
    #     os.makedirs('./assets/do_intervention')
    
    # iter_test = iter(test_dataloader)
    # count = 100
    # for num in tqdm.tqdm(range(count)):
    #     x_batch, y_batch = next(iter_test)
        
    #     if config["cuda"]:
    #         x_batch = x_batch.cuda()
    #         y_batch = y_batch.cuda()
    
    #     mean, logvar, prior_logvar, latent_orig, causal_latent, align_latent, xhat = model([x_batch, y_batch])
        
    #     # using only mean
    #     epsilon = mean
        
    #     # do-intervention
    #     # do_index = 0
    #     # do_value = 0
    #     for do_index in range(config["node"]):
    #         fig, ax = plt.subplots(3, 3, figsize=(5, 5))
            
    #         for k, do_value in enumerate(np.linspace(-0.8, 0.8, 9)):
    #             do_value = round(do_value, 1)
    #             causal_latent_ = [x.clone() for x in causal_latent]
    #             causal_latent_[do_index] = torch.tensor([[do_value] * config["node_dim"]], dtype=torch.float32)
    #             z = model.inverse(causal_latent_)
    #             z = torch.cat(z, dim=0).clone().detach()
    #             for j in range(config["node"]):
    #                 if j == do_index:
    #                     continue
    #                 else:
    #                     if j == 0:  # root node
    #                         z[j, :] = epsilon[:, j]
    #                     z[j, :] = torch.matmul(model.B[:j, j].t(), z[:j, :]) + epsilon[:, j]
    #             z = torch.split(z, 1, dim=0)
    #             z = list(map(lambda x, layer: torch.tanh(layer(x)), z, model.flows))
                
    #             do_xhat = model.decoder(torch.cat(z, dim=1)).view(96, 96, 3)

    #             ax.flatten()[k].imshow((do_xhat.clone().detach().cpu().numpy() + 1) / 2)
    #             ax.flatten()[k].axis('off')
    #             ax.flatten()[k].set_title('x = {}'.format(do_value))
    #             # ax.flatten()[k].set_title('do({} = {})'.format(name[do_index], do_value))
            
    #         plt.suptitle('do({} = x)'.format(name[do_index]), fontsize=15)
    #         plt.savefig('./assets/do_intervention/{}_do_{}.png'.format(num, name[do_index]), bbox_inches='tight')
    #         # plt.show()
    #         plt.close()
            
    wandb.run.finish()
#%%
if __name__ == '__main__':
    main()
#%%