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

from utils.model import (
    VAE,
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
    project="(causal)VAE", 
    entity="anseunghwan",
    tags=["NPSEM", "VQVAE"],
)
#%%
import argparse
def get_args(debug):
    parser = argparse.ArgumentParser('parameters')

    parser.add_argument('--seed', type=int, default=1, 
                        help='seed for repeatable results')
    
    parser.add_argument("--node", default=4, type=int,
                        help="the number of nodes")
    parser.add_argument("--num_embeddings", default=10, type=int,
                        help="the number of embedding vectors")
    parser.add_argument("--embedding_dim", default=2, type=int,
                        help="dimension of embedding vector")
    parser.add_argument("--hidden_dim", default=2, type=int,
                        help="dimension of shared layer")
    
    parser.add_argument('--epochs', default=100, type=int,
                        help='maximum iteration')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='batch size')
    parser.add_argument('--lr', default=0.001, type=float,
                        help='learning rate')
    
    parser.add_argument('--beta', default=0.25, type=float,
                        help='weight of commitment loss')
    parser.add_argument('--gamma', default=1, type=float,
                        help='weight of label alignment loss')
    
    parser.add_argument('--fig_show', default=False, type=bool)

    if debug:
        return parser.parse_args(args=[])
    else:    
        return parser.parse_args()
#%%
def train(dataloader, model, B, config, optimizer, device):
    logs = {
        'loss': [], 
        'recon': [],
        'VQ': [],
        'Align': [],
    }
    
    for (x_batch, y_batch) in tqdm.tqdm(iter(dataloader), desc="inner loop"):
        
        if config["cuda"]:
            x_batch = x_batch.cuda()
            y_batch = y_batch.cuda()
        
        with torch.autograd.set_detect_anomaly(True):    
            optimizer.zero_grad()
            
            causal_latent, xhat, vq_loss, label_hat = model(x_batch)
            
            loss_ = []
            
            """reconstruction"""
            recon = 0.5 * torch.pow(xhat - x_batch, 2).sum(axis=[1, 2, 3]).mean() 
            # recon = F.mse_loss(xhat, batch)
            loss_.append(('recon', recon))

            """VQ loss"""
            loss_.append(('VQ', vq_loss))
            
            """Label Alignment"""
            align_loss = 0.5 * torch.pow(label_hat - y_batch, 2).sum(axis=1).mean() 
            loss_.append(('Align', align_loss))
            
            loss = recon + vq_loss + config["gamma"] * align_loss
            loss_.append(('loss', loss))
            
            loss.backward()
            optimizer.step()
        
        """accumulate losses"""
        for x, y in loss_:
            logs[x] = logs.get(x) + [y.item()]
    
    return logs, B, xhat
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
    B[:2, 2:] = torch.tensor(np.array([[1, 1], [1, 1]]))
    
    model = VAE(B, config, device)
    model.to(device)
    
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config["lr"]
    )
    
    wandb.watch(model, log_freq=100) # tracking gradients
    model.train()
    
    # for epoch in tqdm.tqdm(range(config["epochs"]), desc="optimization for ML"):
    for epoch in range(config["epochs"]):
        logs, B, xhat = train(dataloader, model, B, config, optimizer, device)
        
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
    torch.save(model.state_dict(), './assets/model.pth')
    artifact = wandb.Artifact('model', type='model') # description=""
    artifact.add_file('./assets/model.pth')
    wandb.log_artifact(artifact)
    
    # """model load"""
    # artifact = wandb.use_artifact('anseunghwan/(causal)VAE/model:v1', type='model')
    # model_dir = artifact.download()
    # model = VAE(config)
    # model.load_state_dict(torch.load(model_dir + '/model.pth'))
    
    # """test dataset"""
    # test_imgs = os.listdir('./utils/causal_data/pendulum/test')
    # test_x = []
    # for i in tqdm.tqdm(range(len(test_imgs)), desc="test data loading"):
    #     test_x.append(np.array(Image.open("./utils/causal_data/pendulum/test/{}".format(test_imgs[i])))[:, :, :3])
    # test_x = (np.array(test_x).astype(float) - 127.5) / 127.5
    # test_x = torch.Tensor(test_x) 
    
    # model.eval()
    
    # """intervention"""
    # causal_latent_orig, causal_latent, xhat, vq_loss, label_hat = model(test_x[:config["batch_size"]])
    # plt.imshow((test_x[0].cpu().detach().numpy() + 1) / 2)
    # plt.axis('off')
    # plt.savefig('./assets/original.png')
    # plt.close()
    
    # """reconstruction with intervention"""
    # latent = model.encoder(nn.Flatten()(test_x[:config["batch_size"]]))
    # epsilon, _ = model.vq_layer(latent)
    
    # z = causal_latent_orig.clone().detach()[0]
    # e = epsilon[0]
    # do_index = 1
    # do_value = -100
    # for j in range(config["node"]):
    #     if j == do_index:
    #         z[:, [j]] = do_value
    #     else:
    #         if j == 0:  # root node
    #             z[:, [j]] = e[j, :]
    #         z[:, [j]] = torch.matmul(z[:, :j], torch.tensor(model.B)[:j, [j]].to(device)) + e[j, :]
    # z = torch.tanh(z)
    
    # do_xhat = model.decoder(z.unsqueeze(dim=0)).view(96, 96, 3)
    # plt.imshow((do_xhat.clone().detach().cpu().numpy() + 1) / 2)
    # plt.axis('off')
    # plt.savefig('./assets/do_recon.png')
    # plt.close()
    
    wandb.run.finish()
#%%
if __name__ == '__main__':
    main()
#%%