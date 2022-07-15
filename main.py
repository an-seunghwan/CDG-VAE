#%%
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
#%%
import numpy as np
import pandas as pd
import tqdm
from PIL import Image
import os
import matplotlib.pyplot as plt

import torch

from utils.simulation import (
    set_random_seed,
    is_dag,
)

from utils.viz import (
    viz_graph,
    viz_heatmap,
)

from utils.model import (
    VAE
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
    # tags=[""],
)
#%%
import argparse
def get_args(debug):
    parser = argparse.ArgumentParser('parameters')

    parser.add_argument('--seed', type=int, default=1, 
                        help='seed for repeatable results')
    
    parser.add_argument("--hidden_dim", default=8, type=int,
                        help="hidden dimensions for MLP")
    parser.add_argument("--num_layer", default=5, type=int,
                        help="hidden dimensions for MLP")
    parser.add_argument("--latent_dim", default=4, type=int,
                        help="dimension of each latent node")
    
    parser.add_argument('--epochs', default=50, type=int,
                        help='maximum iteration')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='batch size')
    parser.add_argument('--lr', default=0.001, type=float,
                        help='learning rate')
    
    parser.add_argument('--w_threshold', default=0.2, type=float,
                        help='threshold for weighted adjacency matrix')
    parser.add_argument('--lambda', default=0.1, type=float,
                        help='coefficient of LASSO penalty')
    parser.add_argument('--beta', default=1, type=float,
                        help='coefficient of KL-divergence')
    
    parser.add_argument('--fig_show', default=False, type=bool)

    if debug:
        return parser.parse_args(args=[])
    else:    
        return parser.parse_args()
#%%
def train(train_x, model, config, optimizer):
    model.train()
    
    logs = {
        'loss': [], 
        'recon': [],
        'L1': [],
        'KL': [],
    }
    
    for i in range(len(train_x) // config["batch_size"]):
        idx = np.random.choice(range(len(train_x)), config["batch_size"])
        batch = torch.FloatTensor(train_x[idx])
        batch = batch.permute((0, 3, 1, 2))
        if config["cuda"]:
            batch = batch.cuda()
            
        optimizer.zero_grad()
        
        z, B_trans_z, z_sem, xhat = model(batch)
        
        loss_ = []
        
        """reconstruction"""
        recon = 0.5 * torch.pow(xhat - batch, 2).sum(axis=[1, 2, 3]).mean()
        loss_.append(('recon', recon))

        """KL-divergence"""
        KL = 0.5 * torch.pow(z - B_trans_z, 2).sum(axis=1).mean()
        loss_.append(('KL', KL))
        
        """Sparsity"""
        L1 = torch.linalg.norm(model.W * model.ReLU_Y, ord=1)
        loss_.append(('L1', L1))
        
        loss = recon + config["beta"] * KL + config["lambda"] * L1
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
    wandb.config.update(config)
    
    set_random_seed(config["seed"])
    torch.manual_seed(config["seed"])
    if config["cuda"]:
        torch.cuda.manual_seed(config["seed"])

    train_imgs = os.listdir('./utils/causal_data/pendulum/train')
    test_imgs = os.listdir('./utils/causal_data/pendulum/test')
    train_x = []
    for i in tqdm.tqdm(range(len(train_imgs)), desc="train data loading"):
        train_x.append(np.array(Image.open("./utils/causal_data/pendulum/train/{}".format(train_imgs[i])))[:, :, :3])
    test_x = []
    for i in tqdm.tqdm(range(len(test_imgs)), desc="test data loading"):
        test_x.append(np.array(Image.open("./utils/causal_data/pendulum/test/{}".format(test_imgs[i])))[:, :, :3])
    train_x = np.array(train_x).astype(float) / 255.
    test_x = np.array(test_x).astype(float) / 255.

    # wandb.run.summary['W_true'] = wandb.Table(data=pd.DataFrame(W_true))
    # fig = viz_graph(W_true, size=(7, 7), show=config["fig_show"])
    # wandb.log({'Graph': wandb.Image(fig)})
    # fig = viz_heatmap(W_true, size=(5, 4), show=config["fig_show"])
    # wandb.log({'heatmap': wandb.Image(fig)})
    
    model = VAE(config)

    if config["cuda"]:
        model.cuda()

    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config["lr"]
    )
    
    for epoch in tqdm.tqdm(range(config["epochs"]), desc="optimization for ML"):
        logs, xhat = train(train_x, model, config, optimizer)
        
        """update log"""
        wandb.log({x : np.mean(y) for x, y in logs.items()})
        
        if epoch % 3 == 0:
            plt.figure(figsize=(4, 4))
            for i in range(9):
                plt.subplot(3, 3, i+1)
                plt.imshow(xhat[i].permute((1, 2, 0)).detach().numpy())
                plt.title('{}'.format(i))
                plt.axis('off')
            plt.savefig('./assets/image_{}.png'.format(epoch))
            # plt.show()
            plt.close()
        
    B_est = (model.W * model.ReLU_Y).detach().numpy()
    B_est[np.abs(B_est) < config["w_threshold"]] = 0.
    B_est = B_est.astype(float).round(2)

    fig = viz_graph(B_est, size=(7, 7), show=config["fig_show"])
    wandb.log({'Graph_est': wandb.Image(fig)})
    fig = viz_heatmap(B_est, size=(5, 4), show=config["fig_show"])
    wandb.log({'heatmap_est': wandb.Image(fig)})

    wandb.run.finish()
#%%
if __name__ == '__main__':
    main()
#%%