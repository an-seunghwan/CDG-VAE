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

import torch
from torch.utils.data import TensorDataset, DataLoader

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
    tags=["linear"], # AddictiveNoiseModel, nonlinear(tanh)
)
#%%
import argparse
def get_args(debug):
    parser = argparse.ArgumentParser('parameters')

    parser.add_argument('--seed', type=int, default=1, 
                        help='seed for repeatable results')
    
    # parser.add_argument("--hidden_dim", default=4, type=int,
    #                     help="hidden dimensions for MLP")
    # parser.add_argument("--num_layer", default=5, type=int,
    #                     help="hidden dimensions for MLP")
    parser.add_argument("--latent_dim", default=4, type=int,
                        help="dimension of each latent node")
    
    parser.add_argument('--epochs', default=100, type=int,
                        help='maximum iteration')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='batch size')
    parser.add_argument('--lr', default=0.001, type=float,
                        help='learning rate')
    
    parser.add_argument('--w_threshold', default=0.01, type=float,
                        help='threshold for weighted adjacency matrix')
    parser.add_argument('--lambda', default=1, type=float,
                        help='coefficient of LASSO penalty')
    parser.add_argument('--beta', default=1, type=float,
                        help='coefficient of KL-divergence')
    
    parser.add_argument('--fig_show', default=False, type=bool)

    if debug:
        return parser.parse_args(args=[])
    else:    
        return parser.parse_args()
#%%
def train(dataloader, model, config, optimizer):
    logs = {
        'loss': [], 
        'recon': [],
        'L1': [],
        'KL': [],
    }
    
    # for i in tqdm.tqdm(range(len(train_x) // config["batch_size"]), desc="inner loop"):
    for batch in tqdm.tqdm(iter(dataloader), desc="inner loop"):
        # idx = np.random.choice(range(len(train_x)), config["batch_size"])
        # batch = torch.FloatTensor(train_x[idx])
        # batch = batch.permute((0, 3, 1, 2))
        
        if config["cuda"]:
            batch = batch.cuda()
            
        batch = batch[0]
        
        optimizer.zero_grad()
        
        z, logvar, B_trans_z, z_sem, xhat = model(batch)
        
        loss_ = []
        
        """reconstruction"""
        recon = 0.5 * torch.pow(xhat - batch, 2).sum(axis=[1, 2, 3]).mean()
        # recon = -((batch * torch.log(xhat) + (1. - batch) * torch.log(1. - xhat)).sum(axis=[1, 2, 3]).mean())
        loss_.append(('recon', recon))

        """KL-divergence"""
        KL = torch.pow(z - B_trans_z, 2).sum(axis=1)
        KL += torch.exp(logvar).sum(axis=1)
        KL -= logvar.sum(axis=1)
        KL -= config["latent_dim"]
        KL *= 0.5
        KL = KL.mean()
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
    train_x = []
    for i in tqdm.tqdm(range(len(train_imgs)), desc="train data loading"):
        train_x.append(np.array(Image.open("./utils/causal_data/pendulum/train/{}".format(train_imgs[i])))[:, :, :3])
    train_x = (np.array(train_x).astype(float) - 127.5) / 127.5
    
    # test_imgs = os.listdir('./utils/causal_data/pendulum/test')
    # test_x = []
    # for i in tqdm.tqdm(range(len(test_imgs)), desc="test data loading"):
    #     test_x.append(np.array(Image.open("./utils/causal_data/pendulum/test/{}".format(test_imgs[i])))[:, :, :3])
    # test_x = (np.array(test_x).astype(float) - 127.5) / 127.5
    
    train_x = torch.Tensor(train_x) 
    dataset = TensorDataset(train_x) 
    dataloader = DataLoader(dataset, 
                            batch_size=config["batch_size"],
                            shuffle=True)
    
    # plt.imshow(train_x[0])
    # plt.show()

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
    
    wandb.watch(model, log_freq=50)
    model.train()
    
    # for epoch in tqdm.tqdm(range(config["epochs"]), desc="optimization for ML"):
    for epoch in range(config["epochs"]):
        logs, xhat = train(dataloader, model, config, optimizer)
        
        with torch.no_grad():
            B_ = (model.W * model.ReLU_Y).detach().clone()
            nonzero_ratio = (B_ != 0).sum().item() / (config["latent_dim"] * (config["latent_dim"] - 1) / 2)
            
        print_input = "[epoch {:03d}]".format(epoch + 1)
        print_input += ''.join([', {}: {:.4f}'.format(x, np.mean(y).round(2)) for x, y in logs.items()])
        print_input += ', NonZero: {:.2f}'.format(nonzero_ratio)
        print(print_input)
        
        """update log"""
        wandb.log({x : np.mean(y) for x, y in logs.items()})
        wandb.log({'NonZero' : nonzero_ratio})
        
        if epoch % 10 == 0:
            plt.figure(figsize=(4, 4))
            for i in range(9):
                plt.subplot(3, 3, i+1)
                # plt.imshow(xhat[i].permute((1, 2, 0)).detach().numpy())
                plt.imshow((xhat[i].detach().numpy() + 1) / 2)
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

    """model save"""
    torch.save(model.state_dict(), './assets/model.pth')
    artifact = wandb.Artifact('model', type='model') # description=""
    artifact.add_file('./assets/model.pth')
    wandb.log_artifact(artifact)
    
    """model load"""
    # artifact = wandb.use_artifact('anseunghwan/(causal)VAE/model:v1', type='model')
    # model_dir = artifact.download()
    # model = VAE(config)
    # model.load_state_dict(torch.load(model_dir + '/model.pth'))
    
    wandb.run.finish()
#%%
if __name__ == '__main__':
    main()
#%%