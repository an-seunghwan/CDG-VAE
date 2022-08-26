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

from utils.model_v1 import (
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
    tags=["NPSEM", "Identifiable"],
)
#%%
import argparse
def get_args(debug):
    parser = argparse.ArgumentParser('parameters')

    parser.add_argument('--seed', type=int, default=1, 
                        help='seed for repeatable results')
    
    parser.add_argument("--node", default=4, type=int,
                        help="the number of nodes")
    
    parser.add_argument('--epochs', default=100, type=int,
                        help='maximum iteration')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='batch size')
    parser.add_argument('--lr', default=0.001, type=float,
                        help='learning rate')
    
    parser.add_argument('--beta', default=0.1, type=float,
                        help='observation noise')
    parser.add_argument('--lambda', default=0.1, type=float,
                        help='weight of alignment loss')
    
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
        'KL': [],
        'align': [],
    }
    
    for (x_batch, y_batch) in tqdm.tqdm(iter(dataloader), desc="inner loop"):
        
        if config["cuda"]:
            x_batch = x_batch.cuda()
            y_batch = y_batch.cuda()
        
        with torch.autograd.set_detect_anomaly(True):    
            optimizer.zero_grad()
            
            logvar, prior_logvar, latent, align, xhat = model([x_batch, y_batch])
            
            loss_ = []
            
            """reconstruction"""
            # recon = 0.5 * torch.pow(xhat - x_batch, 2).sum(axis=[1, 2, 3]).mean() 
            recon = F.mse_loss(xhat, x_batch)
            loss_.append(('recon', recon))
            
            """KL-Divergence"""
            KL = 0
            KL += prior_logvar.sum(axis=1)
            KL -= logvar.sum(axis=1)
            KL += torch.exp(logvar - prior_logvar).sum(axis=1)
            KL -= config["node"]
            KL *= 0.5
            KL = KL.mean()
            loss_.append(('KL', KL))
            
            """Label Alignment"""
            align = torch.pow(align - y_batch, 2).sum(axis=1).mean()
            loss_.append(('align', align))
            
            loss = recon + config["beta"] * KL 
            loss += config["lambda"] * align
            loss_.append(('loss', loss))
            
            loss.backward()
            optimizer.step()
        
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
            label = label / label.std(axis=0)
            self.y_data = label.round(2)

        def __len__(self): 
            return len(self.x_data)

        def __getitem__(self, idx): 
            x = torch.FloatTensor(self.x_data[idx])
            y = torch.FloatTensor(self.y_data[idx])
            return x, y
    
    dataset = CustomDataset()
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)
    
    """Estimated Causal Adjacency Matrix"""
    B = torch.zeros(config["node"], config["node"])
    B[:2, 2:] = 1
    
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
        
        if epoch % 3 == 0:
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
    torch.save(model.state_dict(), './assets/model.pth')
    artifact = wandb.Artifact('model', type='model') # description=""
    artifact.add_file('./assets/model.pth')
    wandb.log_artifact(artifact)
    
    # """model load"""
    # artifact = wandb.use_artifact('anseunghwan/(causal)VAE/model:v3', type='model')
    # model_dir = artifact.download()
    # model = VAE(B, config, device).to(device)
    # if config["cuda"]:
    #     model.load_state_dict(torch.load(model_dir + '/model.pth'))
    # else:
    #     model.load_state_dict(torch.load(model_dir + '/model.pth', map_location=torch.device('cpu')))
    
    # """test dataset"""
    # class TestCustomDataset(Dataset): 
    #     def __init__(self):
    #         test_imgs = os.listdir('./utils/causal_data/pendulum/test')
    #         test_x = []
    #         for i in tqdm.tqdm(range(len(test_imgs)), desc="test data loading"):
    #             test_x.append(np.array(
    #                 Image.open("./utils/causal_data/pendulum/test/{}".format(test_imgs[i])).resize((96, 96))
    #                 )[:, :, :3])
    #         self.x_data = (np.array(test_x).astype(float) - 127.5) / 127.5
            
    #         label = np.array([x[:-4].split('_')[1:] for x in test_imgs]).astype(float)
    #         label = label - label.mean(axis=0)
    #         # label = label / label.std(axis=0)
    #         self.y_data = label.round(2)

    #     def __len__(self): 
    #         return len(self.x_data)

    #     def __getitem__(self, idx): 
    #         x = torch.FloatTensor(self.x_data[idx])
    #         y = torch.FloatTensor(self.y_data[idx])
    #         return x, y
    
    # test_dataset = TestCustomDataset()
    # test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # model.eval()
    
    # """intervention"""
    # iter_test = iter(test_dataloader)
    # x_batch, y_batch = next(iter_test)
    # if config["cuda"]:
    #     x_batch = x_batch.cuda()
    #     y_batch = y_batch.cuda()
    
    # logvar, prior_logvar, latent_orig, causal_latent, xhat = model([x_batch, y_batch])
    # plt.imshow((x_batch[0].cpu().detach().numpy() + 1) / 2)
    # plt.axis('off')
    # plt.savefig('./assets/original.png')
    # plt.close()
    
    # plt.imshow((xhat[0].cpu().detach().numpy() + 1) / 2)
    # plt.axis('off')
    # plt.savefig('./assets/recon.png')
    # plt.close()
    
    # """reconstruction with intervention"""
    # # z = torch.cat(causal_latent, dim=0).clone().detach()
    # epsilon = torch.exp(logvar / 2) * torch.randn(config["node"] * config["node_dim"]).to(device) 
    # epsilon = epsilon.view(config["node"], config["node_dim"]).contiguous()
    
    # do_index = 0
    # do_value = 0.9
    
    # causal_latent[do_index] = torch.tensor([[do_value, do_value]])
    # z = model.inverse(causal_latent)
    # z = torch.cat(z, dim=0).clone().detach()
    # for j in range(config["node"]):
    #     if j == 0:  # root node
    #         z[j, :] = epsilon[j, :]
    #     z[j, :] = torch.matmul(model.B[:j, j].t(), z[:j, :]) + epsilon[j, :]
    # z = torch.split(z, 1, dim=0)
    # z = list(map(lambda x, layer: torch.tanh(layer(x)), z, model.flows))
    
    # do_xhat = model.decoder(torch.cat(z, dim=1)).view(96, 96, 3)
    # plt.imshow((do_xhat.clone().detach().cpu().numpy() + 1) / 2)
    # plt.axis('off')
    # plt.savefig('./assets/do_recon.png')
    # plt.close()
    
    wandb.run.finish()
#%%
if __name__ == '__main__':
    main()
#%%