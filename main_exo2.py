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

from utils.model_exo import (
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
    tags=["nonlinear"]
        # "prior_constraint: DAG reconstruction",], 
        # AddictiveNoiseModel, nonlinear(tanh)
)
#%%
import argparse
def get_args(debug):
    parser = argparse.ArgumentParser('parameters')

    parser.add_argument('--seed', type=int, default=1, 
                        help='seed for repeatable results')
    
    parser.add_argument("--latent_dim", default=5, type=int,
                        help="dimension of each latent node")
    
    parser.add_argument('--epochs', default=100, type=int,
                        help='maximum iteration')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='batch size')
    parser.add_argument('--lr', default=0.001, type=float,
                        help='learning rate')
    
    parser.add_argument('--noise', default=0.5, type=float,
                        help='std of exogenous posterior distribution')
    parser.add_argument('--penalty', default='without', type=str,
                        help='penalty type for sparity: without, lasso')
    parser.add_argument('--lambda1', default=0.1, type=float,
                        help='coefficient of sparsity penalty')
    parser.add_argument('--beta', default=5, type=float,
                        help='coefficient of KL-divergence')
    # parser.add_argument('--lambda2', default=0.1, type=float,
    #                     help='coefficient of prior constraint')
    parser.add_argument('--w_threshold', default=0.1, type=float,
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
        # 'prior': [],
        'Sparsity': [],
    }
    
    for batch in tqdm.tqdm(iter(dataloader), desc="inner loop"):
        
        batch = batch[0]
        # batch.to(device)
        if config["cuda"]:
            batch = batch.cuda()
        # break
        
        with torch.autograd.set_detect_anomaly(True):    
            optimizer.zero_grad()
            
            exog, latent, B, xhat = model(batch)
            
            loss_ = []
            
            """reconstruction"""
            recon = 0.5 * torch.pow(xhat - batch, 2).sum(axis=[1, 2, 3]).mean() # Gaussian
            # recon = -((batch * torch.log(xhat) + (1. - batch) * torch.log(1. - xhat)).sum(axis=[1, 2, 3]).mean())
            loss_.append(('recon', recon))

            """KL-divergence"""
            KL = torch.pow(exog, 2).sum(axis=1)
            # KL += exog_logvar.sum(axis=1)
            # KL += torch.exp(exog_logvar).sum(axis=1)
            # KL -= config["latent_dim"]
            KL *= 0.5
            KL = KL.mean()
            loss_.append(('KL', KL))
            
            # """prior constraint""" # DAG reconstruction
            # prior = (0.5 * torch.pow(inverse_sigmoid_latent - torch.matmul(inverse_sigmoid_latent, B), 2).sum(axis=1)).mean()
            # loss_.append(('prior', prior))
            
            """Sparsity"""
            if config["penalty"] == "without":
                loss = recon + config["beta"] * KL
                
            else:
                if config["penalty"] == "lasso":
                    sparsity = torch.linalg.norm(B, ord=1)
                    loss_.append(('Sparsity', sparsity))
                    loss = recon + config["beta"] * KL + config["lambda1"] * sparsity
                    
                elif config["penalty"] == "MCP":
                    p1 = config["lambda"] * torch.abs(B)
                    p1 -= torch.pow(B, 2) / (2. * config["gamma"])
                    p1 = p1[torch.abs(B) <= config["gamma"] * config["lambda"]].sum()
                    
                    p2 = (torch.abs(B) > config["gamma"] * config["lambda"]).sum().float()
                    p2 *= torch.tensor(0.5 * config["gamma"] * (config["lambda"] ** 2))
                    
                    sparsity = p1 + p2
                    loss_.append(('Sparsity', sparsity))
                    loss = recon + config["beta"] * KL + sparsity
                    
                else:
                    raise ValueError("Unknown penalty type.")
            loss_.append(('loss', loss))
            
            # loss = recon + config["beta"] * KL
            # loss_.append(('loss', loss))
            
            loss.backward()
            optimizer.step()
            
            """accumulate losses"""
            for x, y in loss_:
                logs[x] = logs.get(x) + [y.item()]
    
    return logs, B, xhat
#%%
def main():
    config = vars(get_args(debug=False)) # default configuration
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
    
    # plt.imshow(train_x[0])
    # plt.show()

    # wandb.run.summary['W_true'] = wandb.Table(data=pd.DataFrame(W_true))
    # fig = viz_graph(W_true, size=(7, 7), show=config["fig_show"])
    # wandb.log({'Graph': wandb.Image(fig)})
    # fig = viz_heatmap(W_true, size=(5, 4), show=config["fig_show"])
    # wandb.log({'heatmap': wandb.Image(fig)})
    
    model = VAE(config, device)
    
    model.to(device)
    # if config["cuda"]:
    #     model.cuda()

    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config["lr"]
    )
    
    wandb.watch(model, log_freq=50) # tracking gradients
    model.train()
    
    # for epoch in tqdm.tqdm(range(config["epochs"]), desc="optimization for ML"):
    for epoch in range(config["epochs"]):
        logs, B, xhat = train(dataloader, model, config, optimizer, device)
        
        with torch.no_grad():
            # """update mask"""
            # if epoch > config["epochs"] // 3:
            #     B_ = (model.W * model.ReLU_Y).detach().clone()
            #     model.mask[torch.abs(B_) < config["w_threshold"]] = 0. 
            nonzero_ratio = (torch.abs(B) >= config["w_threshold"]).sum().item() 
            nonzero_ratio /= (config["latent_dim"] * (config["latent_dim"] - 1) / 2)
            
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
    
    """post-process"""
    # B_est = (model.W * model.ReLU_Y * model.mask).cpu().detach().numpy()
    # B_est = (model.W * model.ReLU_Y).cpu().detach().numpy()
    # B_est[np.abs(B_est) < config["w_threshold"]] = 0.
    B_est = (torch.sigmoid(model.W / model.tau) * 2 - 1) * model.ReLU_Y # without Gumbel-Sigmoid (-1 to 1)
    B_est = B_est.cpu().detach().numpy()
    B_est = B_est.astype(float).round(2)
    
    nonzero_ratio = (B_est != 0).sum() / (config["latent_dim"] * (config["latent_dim"] - 1) / 2)
    wandb.log({'NonZero' : nonzero_ratio})

    fig = viz_graph(B_est, size=(7, 7), show=config["fig_show"])
    wandb.log({'Graph_est': wandb.Image(fig)})
    fig = viz_heatmap(np.flipud(B_est), size=(5, 4), show=config["fig_show"])
    wandb.log({'heatmap_est': wandb.Image(fig)})

    """model save"""
    torch.save(model.state_dict(), './assets/model.pth')
    artifact = wandb.Artifact('model', type='model') # description=""
    artifact.add_file('./assets/model.pth')
    wandb.log_artifact(artifact)
    
    # """model load"""
    # artifact = wandb.use_artifact('anseunghwan/(causal)VAE/model:v16', type='model')
    # model_dir = artifact.download()
    # model = VAE(config, device)
    # model.load_state_dict(torch.load(model_dir + '/model.pth'))
    
    # B_est = (model.W * model.ReLU_Y).cpu().detach().numpy()
    # B_est[np.abs(B_est) < 0.01] = 0.
    
    # # test dataset
    # test_imgs = os.listdir('./utils/causal_data/pendulum/test')
    # test_x = []
    # for i in tqdm.tqdm(range(len(test_imgs)), desc="test data loading"):
    #     test_x.append(np.array(Image.open("./utils/causal_data/pendulum/test/{}".format(test_imgs[i])))[:, :, :3])
    # test_x = (np.array(test_x).astype(float) - 127.5) / 127.5
    # test_x = torch.Tensor(test_x) 
    
    # """intervention"""
    # exog_mean, exog_logvar, latent, B, xhat1, xhat2 = model(test_x[[1]])
    # plt.imshow((xhat1[0].cpu().detach().numpy() + 1) / 2)
    # plt.axis('off')
    # plt.savefig('./assets/original.png')
    # plt.close()
    
    # """reconstruction with intervention"""
    # z = latent.detach().clone()
    # epsilon = exog_mean + torch.exp(exog_logvar / 2) * torch.randn(exog_mean.shape).to(device)
    # do_index = 1
    # do_value = 5
    # for j in range(config["latent_dim"]):
    #     if j == do_index:
    #         z[:, [j]] = do_value
    #     else:
    #         if j == 0:  # root node
    #             z[:, [j]] = epsilon[:, [j]]
    #         z[:, [j]] = torch.matmul(z[:, :j], torch.tensor(B_est)[:j, [j]].to(device)) + epsilon[:, [j]]
    
    # do_xhat = model.decoder(z).view(96, 96, 3)
    # plt.imshow((do_xhat.cpu().detach().numpy() + 1) / 2)
    # plt.axis('off')
    # plt.savefig('./assets/do_recon.png')
    # plt.close()
    
    # """reconstruction with intervention"""
    # fig, axs = plt.subplots(config["latent_dim"], 11, figsize=(10, 5))
    # for do_index in range(config["latent_dim"]):
    #     z = latent.detach().clone()
    #     epsilon = exog_mean + torch.exp(exog_logvar / 2) * torch.randn(exog_mean.shape).to(device)
        
    #     for i, do_value in enumerate(np.linspace(-20, 20, 11)):
    #         for j in range(config["latent_dim"]):
    #             if j == do_index:
    #                 z[:, [j]] = do_value
    #             else:
    #                 if j == 0:  # root node
    #                     z[:, [j]] = epsilon[:, [j]]
    #                 z[:, [j]] = torch.matmul(z[:, :j], torch.tensor(B_est)[:j, [j]].to(device)) + epsilon[:, [j]]
        
    #         do_xhat = model.decoder(z).view(96, 96, 3)
    #         axs[do_index, i].imshow((do_xhat.cpu().detach().numpy() + 1) / 2)
    #         axs[do_index, i].axis('off')
    #     fig.savefig('./assets/do_recon.png')
    #     plt.close()
    
    wandb.run.finish()
#%%
if __name__ == '__main__':
    main()
#%%