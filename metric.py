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

from modules.simulation import (
    set_random_seed,
    is_dag,
)

from modules.viz import (
    viz_graph,
    viz_heatmap,
)

from modules.datasets import (
    LabeledDataset, 
    UnLabeledDataset,
    TestDataset,
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
    project="CausalDisentangled", 
    entity="anseunghwan",
    tags=["VAEBased", "Metric"],
)
#%%
import argparse
def get_args(debug):
    parser = argparse.ArgumentParser('parameters')
    
    parser.add_argument('--num', type=int, default=0, 
                        help='model version')

    if debug:
        return parser.parse_args(args=[])
    else:    
        return parser.parse_args()
#%%
def main():
    #%%
    config = vars(get_args(debug=False)) # default configuration
    
    # model_name = 'VAE'
    # model_name = 'InfoMax'
    model_name = 'CDGVAE'
    # model_name = 'CDGVAEsemi'
    
    scm = 'linear'
    # scm = 'nonlinear'
    
    """model load"""
    artifact = wandb.use_artifact('anseunghwan/CausalDisentangled/model_{}_{}:v{}'.format(model_name, scm, config["num"]), type='model')
    for key, item in artifact.metadata.items():
        config[key] = item
    assert model_name == config["model"]
    assert scm == config["scm"]
    model_dir = artifact.download()
    
    config["cuda"] = torch.cuda.is_available()
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    wandb.config.update(config)
    
    set_random_seed(config["seed"])
    torch.manual_seed(config["seed"])
    if config["cuda"]:
        torch.cuda.manual_seed(config["seed"])

    """dataset"""
    dataset = LabeledDataset(config, downstream=False)

    """
    Causal Adjacency Matrix
    light -> length
    light -> position
    angle -> length
    angle -> position
    """
    B = torch.zeros(config["node"], config["node"])
    B[dataset.name.index('light'), dataset.name.index('length')] = 1
    B[dataset.name.index('light'), dataset.name.index('position')] = 1
    B[dataset.name.index('angle'), dataset.name.index('length')] = 1
    B[dataset.name.index('angle'), dataset.name.index('position')] = 1
    
    """adjacency matrix scaling"""
    if config["adjacency_scaling"]:
        indegree = B.sum(axis=0)
        mask = (indegree != 0)
        B[:, mask] = B[:, mask] / indegree[mask]
    
    """import model"""
    if config["model"] == 'VAE':
        from modules.model import VAE
        model = VAE(B, config, device) 
        
    elif config["model"] == 'InfoMax':
        from modules.model import VAE
        model = VAE(B, config, device) 
        
    elif config["model"] in ['GAM', 'GAMsemi']:
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
        
        from modules.model import GAM
        model = GAM(B, mask, config, device) 
    
    else:
        raise ValueError('Not supported model!')
        
    model = model.to(device)
    
    if config["cuda"]:
        model.load_state_dict(torch.load(model_dir + '/model_{}_{}.pth'.format(config["model"], config["scm"])))
    else:
        model.load_state_dict(torch.load(model_dir + '/model_{}_{}.pth'.format(config["model"], config["scm"]), 
                                         map_location=torch.device('cpu')))
    
    model.eval()
    #%%
    """import baseline classifier"""
    artifact = wandb.use_artifact('anseunghwan/CausalDisentangled/CDMClassifier:v{}'.format(0), type='model')
    model_dir = artifact.download()
    from modules.model import Classifier
    """masking"""
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
        
    classifier = Classifier(mask, config, device) 
    if config["cuda"]:
        classifier.load_state_dict(torch.load(model_dir + '/CDMClassifier.pth'))
    else:
        classifier.load_state_dict(torch.load(model_dir + '/CDMClassifier.pth', map_location=torch.device('cpu')))
    #%%
    """latent range"""
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)
    latents = []
    for x_batch, y_batch in tqdm.tqdm(iter(dataloader)):
        if config["cuda"]:
            x_batch = x_batch.cuda()
            y_batch = y_batch.cuda()
        
        with torch.no_grad():
            if config["model"] in ['GAM', 'GAMsemi']:
                _, _, _, _, latent, _, _, _, _ = model(x_batch, deterministic=True)
            else:
                _, _, _, _, latent, _, _, _ = model(x_batch, deterministic=True)
        latents.append(torch.cat(latent, dim=1))
    
    latents = torch.cat(latents, dim=0)
    
    latent_min = latents.cpu().numpy().min(axis=0)
    latent_max = latents.cpu().numpy().max(axis=0)
    #%%
    """metric"""
    CDM_dict_lower = {x:[] for x in dataset.name}
    CDM_dict_upper = {x:[] for x in dataset.name}
    for s in ['light', 'angle', 'length', 'position']:
        for c in ['light', 'angle', 'length', 'position']:
            CDM_lower = 0
            CDM_upper = 0
            
            dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False)
            for x_batch, y_batch in tqdm.tqdm(iter(dataloader), desc="{} | {}".format(c, s)):
                if config["cuda"]:
                    x_batch = x_batch.cuda()
                    y_batch = y_batch.cuda()

                with torch.no_grad():
                    if config["model"] in ['GAM', 'GAMsemi']:
                        # mean, logvar, epsilon, orig_latent, latent, logdet, align_latent, xhat_separated, xhat = model(x_batch, deterministic=True)
                        mean, logvar, epsilon, orig_latent, latent, logdet = model.encode(x_batch, deterministic=True)
                    else:
                        # mean, logvar, epsilon, orig_latent, latent, logdet, align_latent, xhat = model(x_batch, deterministic=True)
                        mean, logvar, epsilon, orig_latent, latent, logdet = model.encode(x_batch, deterministic=True)

                    do_index = dataset.name.index(s)
                    min_ = latent_min[do_index]
                    max_ = latent_max[do_index]
                    
                    score = []
                    for val in [min_, max_]:
                        latent_ = [x.clone() for x in latent]
                        latent_[do_index] = torch.tensor(val).view(1, 1).repeat(latent[0].size(0), 1).to(device)
                        z = model.inverse(latent_)
                        z = torch.cat(z, dim=1).clone()
                        
                        for j in range(config["node"]):
                            if j == do_index:
                                continue
                            else:
                                if j == 0:  # root node
                                    z[:, j] = epsilon[:, j]
                                z[:, j] = torch.matmul(z[:, :j], model.B[:j, j]) + epsilon[:, j]
                        
                        z = torch.split(z, 1, dim=1)
                        z = list(map(lambda x, layer: layer(x), z, model.flows))
                        z = [z_[0] for z_ in z]
                        
                        if config["model"] in ['GAM', 'GAMsemi']:
                            _, do_xhat = model.decode(z)
                        else:
                            do_xhat = model.decoder(torch.cat(z, dim=1)).view(-1, config["image_size"], config["image_size"], 3)

                        """factor classification"""
                        score.append(torch.sigmoid(classifier(do_xhat))[:, dataset.name.index(c)])
                        
                    CDM_lower += (score[0] - score[1]).sum()
                    CDM_upper += (score[0] - score[1]).abs().sum()
                    
            CDM_lower /= dataset.__len__()
            CDM_upper /= dataset.__len__()
            CDM_dict_lower[s] = CDM_dict_lower.get(s) + [(c, CDM_lower.abs().item())]
            CDM_dict_upper[s] = CDM_dict_upper.get(s) + [(c, CDM_upper.item())]
    #%%
    CDM_mat_lower = np.zeros((config["node"], config["node"]))
    for i, s in enumerate(dataset.name[:4]):
        CDM_mat_lower[i, :] = [x[1] for x in CDM_dict_lower[s]]
    CDM_mat_upper = np.zeros((config["node"], config["node"]))
    for i, s in enumerate(dataset.name[:4]):
        CDM_mat_upper[i, :] = [x[1] for x in CDM_dict_upper[s]]
    
    fig = viz_heatmap(np.flipud(CDM_mat_lower), size=(7, 7))
    wandb.log({'CDM(lower)': wandb.Image(fig)})
    fig = viz_heatmap(np.flipud(CDM_mat_upper), size=(7, 7))
    wandb.log({'CDM(upper)': wandb.Image(fig)})
    
    if not os.path.exists('./assets/CDM/'): 
        os.makedirs('./assets/CDM/')
    # save as csv
    df = pd.DataFrame(CDM_mat_lower.round(3), columns=dataset.name[:4], index=dataset.name[:4])
    df.to_csv('./assets/CDM/lower_{}_{}_{}.csv'.format(config["model"], config["scm"], config['num']))
    df = pd.DataFrame(CDM_mat_upper.round(3), columns=dataset.name[:4], index=dataset.name[:4])
    df.to_csv('./assets/CDM/upper_{}_{}_{}.csv'.format(config["model"], config["scm"], config['num']))
    #%%
    wandb.run.finish()
#%%
if __name__ == '__main__':
    main()
#%%