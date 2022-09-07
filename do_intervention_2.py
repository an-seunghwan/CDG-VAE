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

# from utils.model_0 import (
#     VAE,
# )
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
    
    parser.add_argument('--version', type=int, default=2, 
                        help='model version')
    parser.add_argument('--num', type=int, default=23, 
                        help='model version')

    if debug:
        return parser.parse_args(args=[])
    else:    
        return parser.parse_args()
#%%
def main():
    config = vars(get_args(debug=False)) # default configuration
    
    """model load"""
    artifact = wandb.use_artifact('anseunghwan/(causal)VAE/model_{}:v{}'.format(config["version"], config["num"]), type='model')
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
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

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
    
    """import model"""
    tmp = __import__("utils.model_{}".format(config["version"]), 
                    fromlist=["utils.model_{}".format(config["version"])])
    VAE = getattr(tmp, "VAE")
    
    model = VAE(B, config, device).to(device)
    if config["cuda"]:
        model.load_state_dict(torch.load(model_dir + '/model_{}.pth'.format(config["version"])))
    else:
        model.load_state_dict(torch.load(model_dir + '/model_{}.pth'.format(config["version"]), map_location=torch.device('cpu')))
    
    model.eval()
    
    """causal adjacency matrix"""
    B_est = (model.W * model.mask).detach().cpu().numpy()
    fig = viz_heatmap(np.flipud(B_est), size=(7, 7))
    wandb.log({'B_est': wandb.Image(fig)})
    B = model.W * model.mask
    
    """get intervention range"""
    latents = []
    iter_test = iter(test_dataloader)
    for x_batch, y_batch in tqdm.tqdm(iter_test):
        if config["cuda"]:
            x_batch = x_batch.cuda()
            y_batch = y_batch.cuda()
            
        mean, logvar, orig_latent, latent, align_latent, xhat = model(x_batch, running=False)
        latents.append(torch.cat(latent, dim=1))
        
    latents = torch.cat(latents, dim=0)
    causal_min = np.quantile(latents.detach().numpy(), q=0.05, axis=0)
    causal_max = np.quantile(latents.detach().numpy(), q=0.95, axis=0)
    # causal_min = torch.min(causal_latents, axis=0).values.detach().numpy()
    # causal_max = torch.max(causal_latents, axis=0).values.detach().numpy()
    
    # h = model.encoder(nn.Flatten()(x_batch)) # [batch, node * node_dim * 2]
    # mean, logvar = torch.split(h, model.config["node"] * model.config["node_dim"], dim=1)
    # # mean.detach().numpy().round(2)
    # # logvar.detach().numpy().round(2)
    # # torch.exp(logvar)
    
    # I_B_inv = torch.inverse(model.I - model.W * model.mask)
    
    # """Latent Generating Process"""
    # noise = torch.randn(x_batch.size(0), model.config["node"] * model.config["node_dim"]).to(model.device) 
    # epsilon = mean + torch.exp(logvar / 2) * noise
    # epsilon = epsilon.view(-1, model.config["node_dim"], model.config["node"]).contiguous()
    # latent = torch.matmul(epsilon, I_B_inv) # [batch, node_dim, node]
    
    # epsilon.squeeze(1).detach().numpy().round(2).mean(axis=0)
    # latent.squeeze(1).detach().numpy().round(2).mean(axis=0)
    
    # orig_latent = latent.clone()
    # latent = [x.squeeze(dim=2) for x in torch.split(latent, 1, dim=2)] # [batch, node_dim] x node
    # latent = list(map(lambda x, layer: layer(x), latent, model.flows)) # [batch, node_dim] x node
    
    # torch.sigmoid(torch.cat(latent, dim=1)).detach().numpy().round(2)
    # latent[-1]
    
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
    
    # """check exogenous"""
    # unique_root = np.unique(test_dataset.y_data[:, 0])
    # unique_root_idx = np.where(test_dataset.y_data[:, 0] == unique_root[0])[0]
    # x_samples = torch.tensor(test_dataset.x_data[unique_root_idx, ...], dtype=torch.float32)
    # y_samples = torch.tensor(test_dataset.y_data[unique_root_idx, ...], dtype=torch.float32)
    # mean, logvar, orig_latent, latent, align_latent, xhat = model(x_samples)
    
    # plt.plot(mean.detach().numpy()[:, 0])
    
    # torch.pow(mean, 2)
    # torch.exp(logvar)
    
    """reconstruction"""
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    iter_test = iter(test_dataloader)
    count = 2
    for _ in range(count):
        x_batch, y_batch = next(iter_test)
    if config["cuda"]:
        x_batch = x_batch.cuda()
        y_batch = y_batch.cuda()
    
    mean, logvar, orig_latent, latent, align_latent, xhat = model(x_batch, running=False)
    
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
    
    """reconstruction with do-intervention"""
    for do_index, (min, max) in enumerate(zip(causal_min, causal_max)):
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
    
    """for specific example"""
    iter_test = iter(test_dataloader)
    count = 3
    for _ in tqdm.tqdm(range(count)):
        x_batch, y_batch = next(iter_test)
    
    if config["cuda"]:
        x_batch = x_batch.cuda()
        y_batch = y_batch.cuda()
    
    mean, logvar, orig_latent, latent, align_latent, xhat = model(x_batch)
    
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
    causal_value = np.quantile(latents.detach().numpy(), q=0.95, axis=0)
    for k, (do_index, do_value) in enumerate(zip(range(config["node"]), causal_value)):
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
        
        do_xhat = model.decoder(torch.cat(z, dim=1)).view(config["image_size"], config["image_size"], 3)

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