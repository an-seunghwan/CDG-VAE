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

from utils.model_v3 import (
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
    
    parser.add_argument('--seed', type=int, default=1, 
                        help='seed for repeatable results')
    
    parser.add_argument('--version', type=int, default=7, 
                        help='model version')

    parser.add_argument("--node", default=4, type=int,
                        help="the number of nodes")
    parser.add_argument("--node_dim", default=1, type=int,
                        help="dimension of each node")
    
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
        def __init__(self):
            train_imgs = os.listdir('./utils/causal_data/pendulum/train')
            train_x = []
            for i in tqdm.tqdm(range(len(train_imgs)), desc="train data loading"):
                train_x.append(np.array(
                    Image.open("./utils/causal_data/pendulum/train/{}".format(train_imgs[i])).resize((96, 96))
                    )[:, :, :3])
            self.x_data = (np.array(train_x).astype(float) - 127.5) / 127.5
            
            label = np.array([x[:-4].split('_')[1:] for x in train_imgs]).astype(float)
            # min-max scaling
            label = label + np.abs(label.min(axis=0))
            label = label / (label.max(axis=0) + np.abs(label.min(axis=0)))
            self.y_data = label.round(2)

        def __len__(self): 
            return len(self.x_data)

        def __getitem__(self, idx): 
            x = torch.FloatTensor(self.x_data[idx])
            y = torch.FloatTensor(self.y_data[idx])
            return x, y
    
    dataset = CustomDataset()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    name = ['light', 'angle', 'length', 'position']
    
    """Estimated Causal Adjacency Matrix"""
    B = torch.zeros(config["node"], config["node"])
    B[:2, 2:] = 1
    
    """test dataset"""
    class TestCustomDataset(Dataset): 
        def __init__(self):
            test_imgs = os.listdir('./utils/causal_data/pendulum/test')
            test_x = []
            for i in tqdm.tqdm(range(len(test_imgs)), desc="test data loading"):
                test_x.append(np.array(
                    Image.open("./utils/causal_data/pendulum/test/{}".format(test_imgs[i])).resize((96, 96))
                    )[:, :, :3])
            self.x_data = (np.array(test_x).astype(float) - 127.5) / 127.5
            
            label = np.array([x[:-4].split('_')[1:] for x in test_imgs]).astype(float)
            # min-max scaling
            label = label + np.abs(label.min(axis=0))
            label = label / (label.max(axis=0) + np.abs(label.min(axis=0)))
            self.y_data = label.round(2)

        def __len__(self): 
            return len(self.x_data)

        def __getitem__(self, idx): 
            x = torch.FloatTensor(self.x_data[idx])
            y = torch.FloatTensor(self.y_data[idx])
            return x, y
    
    test_dataset = TestCustomDataset()
    test_dataloader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)
    
    """model load"""
    artifact = wandb.use_artifact('anseunghwan/(causal)VAE/model_v3:v{}'.format(config["version"]), type='model')
    model_dir = artifact.download()
    model = VAE(B, config, device).to(device)
    if config["cuda"]:
        model.load_state_dict(torch.load(model_dir + '/model_v3.pth'))
    else:
        model.load_state_dict(torch.load(model_dir + '/model_v3.pth', map_location=torch.device('cpu')))
    
    model.eval()
    
    """intervention"""
    iter_test = iter(test_dataloader)
    count = 1
    for _ in range(count):
        x_batch, y_batch = next(iter_test)
    if config["cuda"]:
        x_batch = x_batch.cuda()
        y_batch = y_batch.cuda()
    
    mean, logvar, prior_logvar, latent1, latent2, latent3, align_latent, xhat = model([x_batch, y_batch])
    
    """reconstruction with do-intervention"""
    # using only mean
    epsilon = mean
    # noise = torch.randn(1, config["node"] * config["node_dim"]).to(device) 
    # epsilon = mean + torch.exp(logvar / 2) * noise
    # epsilon = epsilon.view(config["node"], config["node_dim"]).contiguous()
    
    fig, ax = plt.subplots(1, 2, figsize=(4, 2))
    
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
    
    # do-intervention
    # do_index = 1
    # do_value = torch.atanh(torch.tensor(0))
    for do_index in range(config["node"]):
        fig, ax = plt.subplots(3, 3, figsize=(5, 5))
        
        for k, do_value in enumerate(np.linspace(-0.8, 0.8, 9)):
            do_value = round(do_value, 1)
            do_value_ = torch.atanh(torch.tensor(do_value, dtype=torch.float32))
            
            z = torch.zeros(mean.size(0), config["node"] * config["node_dim"])
            for j in range(config["node"]):
                child = [j + i * config["node"] for i in range(config["node_dim"])]
                if j == do_index:
                    z[:, child] = do_value_
                else:
                    h = model.lgp_nets[j](model.B[:, j].repeat(config["node_dim"]) * z)
                    z[:, child] = h + epsilon[:, child]
            z = torch.tanh(z)
            do_xhat = model.decoder(z).view(mean.size(0), 96, 96, 3)[0]

            ax.flatten()[k].imshow((do_xhat.clone().detach().cpu().numpy() + 1) / 2)
            ax.flatten()[k].axis('off')
            ax.flatten()[k].set_title('x = {}'.format(do_value))
            # ax.flatten()[k].set_title('do({} = {})'.format(name[do_index], do_value))
        
        plt.suptitle('do({} = x)'.format(name[do_index]), fontsize=15)
        plt.savefig('{}/do_{}.png'.format(model_dir, name[do_index]), bbox_inches='tight')
        # plt.show()
        plt.close()
        
        wandb.log({'do intervention on {}'.format(name[do_index]): wandb.Image(fig)})
    
    """for specific example"""
    iter_test = iter(test_dataloader)
    count = 10
    for _ in tqdm.tqdm(range(count)):
        x_batch, y_batch = next(iter_test)
    
    if config["cuda"]:
        x_batch = x_batch.cuda()
        y_batch = y_batch.cuda()
    
    mean, logvar, prior_logvar, latent1, latent2, latent3, align_latent, xhat = model([x_batch, y_batch])
    
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
    for k, (do_index, do_value) in enumerate([(0, 0.8), (1, -0.5), (2, -0.9), (3, 0.3)]):
        do_value = round(do_value, 1)
        do_value_ = torch.atanh(torch.tensor(do_value, dtype=torch.float32))
        
        z = torch.zeros(mean.size(0), config["node"] * config["node_dim"])
        for j in range(config["node"]):
            child = [j + i * config["node"] for i in range(config["node_dim"])]
            if j == do_index:
                z[:, child] = do_value_
            else:
                h = model.lgp_nets[j](model.B[:, j].repeat(config["node_dim"]) * z)
                z[:, child] = h + epsilon[:, child]
        z = torch.tanh(z)
        do_xhat = model.decoder(z).view(mean.size(0), 96, 96, 3)[0]

        ax.flatten()[k+2].imshow((do_xhat.clone().detach().cpu().numpy() + 1) / 2)
        ax.flatten()[k+2].axis('off')
        ax.flatten()[k+2].set_title('do({} = {})'.format(name[do_index], do_value))
    
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