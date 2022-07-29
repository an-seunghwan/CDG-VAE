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
from torch.utils.data import TensorDataset, DataLoader

from utils.simulation import (
    set_random_seed,
    is_dag,
)

from utils.viz import (
    viz_graph,
    viz_heatmap,
)

from utils.model_vanilla import (
    VAE
)
#%%
"""
wandb artifact cache cleanup "1GB"
"""
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
    tags=["vanilla"]
)
#%%
import argparse
def get_args(debug):
    parser = argparse.ArgumentParser('parameters')

    parser.add_argument('--seed', type=int, default=1, 
                        help='seed for repeatable results')
    
    parser.add_argument("--latent_dim", default=2, type=int,
                        help="dimension of each latent node")
    
    parser.add_argument('--epochs', default=100, type=int,
                        help='maximum iteration')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='batch size')
    parser.add_argument('--lr', default=0.001, type=float,
                        help='learning rate')
    
    parser.add_argument('--beta', default=5, type=float,
                        help='coefficient of KL-divergence')
    parser.add_argument('--lambda2', default=1, type=float,
                        help='threshold for adjacency matrix')
    
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
    }
    
    for batch in tqdm.tqdm(iter(dataloader), desc="inner loop"):
        
        batch = batch[0]
        if config["cuda"]:
            batch = batch.cuda()
        # break
        
        with torch.autograd.set_detect_anomaly(True):    
            optimizer.zero_grad()
            
            exog_mean, exog_logvar, latent, xhat = model(batch)
            
            loss_ = []
            
            """reconstruction"""
            recon = 0.5 * torch.pow(xhat - batch, 2).sum(axis=[1, 2, 3]).mean() # Gaussian
            loss_.append(('recon', recon))

            """KL-divergence"""
            KL = torch.pow(exog_mean, 2).sum(axis=1)
            KL -= exog_logvar.sum(axis=1)
            KL += torch.exp(exog_logvar).sum(axis=1)
            KL -= config["latent_dim"]
            KL *= 0.5
            KL = KL.mean()
            loss_.append(('KL', KL))
            
            loss = recon + config["beta"] * KL
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

    train_imgs = os.listdir('./utils/causal_data/pendulum/train')
    train_x = []
    for i in tqdm.tqdm(range(len(train_imgs)), desc="train data loading"):
        train_x.append(np.array(Image.open("./utils/causal_data/pendulum/train/{}".format(train_imgs[i])))[:, :, :3])
    train_x = (np.array(train_x).astype(float) - 127.5) / 127.5
    
    train_x = torch.Tensor(train_x) 
    dataset = TensorDataset(train_x) 
    dataloader = DataLoader(dataset, 
                            batch_size=config["batch_size"],
                            shuffle=True)
    del train_imgs
    del train_x
    del dataset
    
    model = VAE(config, device)
    
    model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config["lr"]
    )
    
    wandb.watch(model, log_freq=50) # tracking gradients
    model.train()
    
    for epoch in range(config["epochs"]):
        logs, xhat = train(dataloader, model, config, optimizer, device)
        
        print_input = "[epoch {:03d}]".format(epoch + 1)
        print_input += ''.join([', {}: {:.4f}'.format(x, np.mean(y).round(2)) for x, y in logs.items()])
        print(print_input)
        
        """update log"""
        wandb.log({x : np.mean(y) for x, y in logs.items()})
        
        if epoch % 10 == 0:
            plt.figure(figsize=(4, 4))
            for i in range(9):
                plt.subplot(3, 3, i+1)
                # plt.imshow(xhat[i].permute((1, 2, 0)).detach().numpy())
                plt.imshow((xhat[i].cpu().detach().numpy() + 1) / 2)
                plt.axis('off')
            plt.savefig('./assets/tmp_image_{}.png'.format(epoch))
            plt.close()
    
    """reconstruction result"""
    fig = plt.figure(figsize=(4, 4))
    for i in range(9):
        plt.subplot(3, 3, i+1)
        # plt.imshow(xhat[i].permute((1, 2, 0)).detach().numpy())
        plt.imshow((xhat[i].cpu().detach().numpy() + 1) / 2)
        plt.axis('off')
    plt.savefig('./assets/image.png')
    plt.close()
    wandb.log({'reconstruction': wandb.Image(fig)})
    
    """model save"""
    torch.save(model.state_dict(), './assets/vanilla_model.pth')
    artifact = wandb.Artifact('vanilla_model', type='model') # description=""
    artifact.add_file('./assets/vanilla_model.pth')
    wandb.log_artifact(artifact)
    
    # """model load"""
    # artifact = wandb.use_artifact('anseunghwan/(causal)VAE/model:v16', type='model')
    # model_dir = artifact.download()
    # model = VAE(config, device)
    # model.load_state_dict(torch.load(model_dir + '/model.pth'))
    
    # test dataset
    test_imgs = os.listdir('./utils/causal_data/pendulum/test')
    test_x = []
    for i in tqdm.tqdm(range(len(test_imgs)), desc="test data loading"):
        test_x.append(np.array(Image.open("./utils/causal_data/pendulum/test/{}".format(test_imgs[i])))[:, :, :3])
    test_x = (np.array(test_x).astype(float) - 127.5) / 127.5
    test_x = torch.Tensor(test_x)
    if config["cuda"]:
        test_x = test_x.cuda()
    
    exog_mean, exog_logvar, latent, xhat = model(test_x)
    latent = latent.detach().cpu().numpy()
    
    plt.scatter(latent[:, 0], latent[:, 1])
    plt.savefig('./assets/disen_latent.png')
    plt.close()
    
    # from sklearn.decomposition import PCA
    # from sklearn.manifold import TSNE

    # pca = PCA(n_components = 2)
    # pc = pca.fit_transform(pd.DataFrame(latent))
    # tsne = TSNE(n_components = 2).fit_transform(pd.DataFrame(latent))
    
    # # plt.scatter(pc[:, 0], pc[:, 1])
    # plt.scatter(tsne[:, 0], tsne[:, 1])
    # # plt.scatter(latent[:, 0], latent[:, 3])
    # plt.savefig('./assets/disen_latent.png')
    # plt.close()
    
    wandb.run.finish()
#%%
if __name__ == '__main__':
    main()
#%%