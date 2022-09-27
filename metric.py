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
    project="(proposal)CausalVAE", 
    entity="anseunghwan",
    tags=["Metric"],
)
#%%
import argparse
def get_args(debug):
    parser = argparse.ArgumentParser('parameters')
    
    parser.add_argument('--num', type=int, default=16, 
                        help='model version')

    if debug:
        return parser.parse_args(args=[])
    else:    
        return parser.parse_args()
#%%
def main():
    #%%
    config = vars(get_args(debug=True)) # default configuration
    
    # postfix = 'vanilla' # 9
    # postfix = 'InfoMax' # 13
    postfix = 'gam' # 16
    
    """model load"""
    try:
        artifact = wandb.use_artifact('anseunghwan/(proposal)CausalVAE/model_{}:v{}'.format(postfix, config["num"]), type='model')
    except:
        artifact = wandb.use_artifact('anseunghwan/(proposal)CausalVAE/model_{}:v{}'.format(postfix.lower(), config["num"]), type='model')
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
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False)

    # """test dataset"""
    # class TestCustomDataset(Dataset): 
    #     def __init__(self, config):
    #         test_imgs = [x for x in os.listdir('./utils/causal_data/pendulum/test') if x.endswith('png')]
    #         test_x = []
    #         for i in tqdm.tqdm(range(len(test_imgs)), desc="test data loading"):
    #             test_x.append(np.array(
    #                 Image.open("./utils/causal_data/pendulum/test/{}".format(test_imgs[i])).resize((config["image_size"], config["image_size"]))
    #                 )[:, :, :3])
    #         self.x_data = (np.array(test_x).astype(float) - 127.5) / 127.5
            
    #         label = np.array([x[:-4].split('_')[1:] for x in test_imgs]).astype(float)
    #         label = label - label.mean(axis=0)
    #         self.std = label.std(axis=0)
    #         """bounded label: normalize to (0, 1)"""
    #         if config["label_normalization"]: 
    #             label = (label - label.min(axis=0)) / (label.max(axis=0) - label.min(axis=0))
    #         self.y_data = label
    #         self.name = ['light', 'angle', 'length', 'position']

    #     def __len__(self): 
    #         return len(self.x_data)

    #     def __getitem__(self, idx): 
    #         x = torch.FloatTensor(self.x_data[idx])
    #         y = torch.FloatTensor(self.y_data[idx])
    #         return x, y
    
    # test_dataset = TestCustomDataset(config)
    # test_dataloader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)
    
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
    if postfix == 'gam':
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
        
        tmp = __import__("utils.model_{}".format(postfix), 
                        fromlist=["utils.model_{}".format(postfix)])
        VAE = getattr(tmp, "VAE")
        model = VAE(B, mask, config, device).to(device)
    else:
        from utils.model_base import VAE
        model = VAE(B, config, device).to(device)
    
    if config["cuda"]:
        model.load_state_dict(torch.load(model_dir + '/model_{}.pth'.format(postfix)))
    else:
        model.load_state_dict(torch.load(model_dir + '/model_{}.pth'.format(postfix), map_location=torch.device('cpu')))
    
    model.eval()
    #%%
    """import baseline classifier"""
    artifact = wandb.use_artifact('anseunghwan/(proposal)CausalVAE/model_classifier:v{}'.format(0), type='model')
    model_dir = artifact.download()
    from utils.model_classifier import Classifier
    """masking"""
    # if config["dataset"] == 'pendulum':
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
        
    # elif config["dataset"] == 'celeba':
    #     raise NotImplementedError('Not yet for CELEBA dataset!')
    
    classifier = Classifier(mask, config, device) 
    if config["cuda"]:
        classifier.load_state_dict(torch.load(model_dir + '/model_{}.pth'.format('classifier')))
    else:
        classifier.load_state_dict(torch.load(model_dir + '/model_{}.pth'.format('classifier'), map_location=torch.device('cpu')))
    #%%
    """latent range"""
    latents = []
    for x_batch, y_batch in tqdm.tqdm(iter(dataloader)):
        if config["cuda"]:
            x_batch = x_batch.cuda()
            y_batch = y_batch.cuda()
        
        with torch.no_grad():
            if postfix == 'gam':    
                _, _, _, _, latent, _, _, _, _ = model(x_batch, deterministic=True)
            else:
                _, _, _, _, latent, _, _, _ = model(x_batch, deterministic=True)
        latents.append(torch.cat(latent, dim=1))
    
    latents = torch.cat(latents, dim=0)
    
    latent_min = latents.numpy().min(axis=0)
    latent_max = latents.numpy().max(axis=0)
    #%%
    """metric"""
    ACE_dict_lower = {x:[] for x in dataset.name}
    ACE_dict_upper = {x:[] for x in dataset.name}
    s = 'length'
    c = 'light'
    for s in ['light', 'angle', 'length', 'position']:
        for c in ['light', 'angle', 'length', 'position']:
            ACE_lower = 0
            ACE_upper = 0
            
            dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False)
            for x_batch, y_batch in tqdm.tqdm(iter(dataloader)):
                if config["cuda"]:
                    x_batch = x_batch.cuda()
                    y_batch = y_batch.cuda()

                with torch.no_grad():
                    if postfix == 'gam':    
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
                        latent_[do_index] = torch.tensor(val).view(1, 1).repeat(latent[0].size(0), 1)
                        z = model.inverse(latent_)
                        z = torch.cat(z, dim=1).clone()
                        
                        for j in range(config["node"]):
                            if j == do_index:
                                continue
                            else:
                                if j == 0:  # root node
                                    z[:, j] = epsilon[:, j]
                                z[:, j] = torch.matmul(z[:, :j], B[:j, j]) + epsilon[:, j]
                        
                        z = torch.split(z, 1, dim=1)
                        z = list(map(lambda x, layer: layer(x), z, model.flows))
                        z = [z_[0] for z_ in z]
                        
                        if postfix == 'gam':
                            _, do_xhat = model.decode(z)
                        else:
                            do_xhat = model.decoder(torch.cat(z, dim=1)).view(-1, config["image_size"], config["image_size"], 3)

                        """factor classification"""
                        score.append(torch.sigmoid(classifier(do_xhat))[:, dataset.name.index(c)])
                        
                    ACE_lower += (score[0] - score[1]).sum()
                    ACE_upper += (score[0] - score[1]).abs().sum()
                    
            ACE_lower /= dataset.__len__()
            ACE_upper /= dataset.__len__()
            ACE_dict_lower[s] = ACE_dict_lower.get(s) + [(c, ACE_lower.abs().item())]
            ACE_dict_upper[s] = ACE_dict_upper.get(s) + [(c, ACE_upper.item())]
    #%%
    ACE_mat_lower = np.zeros((config["node"], config["node"]))
    for i, c in enumerate(dataset.name):
        ACE_mat_lower[i, :] = [x[1] for x in ACE_dict_lower[c]]
    ACE_mat_upper = np.zeros((config["node"], config["node"]))
    for i, c in enumerate(dataset.name):
        ACE_mat_upper[i, :] = [x[1] for x in ACE_dict_upper[c]]
    
    fig = viz_heatmap(np.flipud(ACE_mat_lower), size=(7, 7))
    wandb.log({'ACE(lower)': wandb.Image(fig)})
    fig = viz_heatmap(np.flipud(ACE_mat_upper), size=(7, 7))
    wandb.log({'ACE(upper)': wandb.Image(fig)})
    
    # save as csv
    pd.DataFrame(ACE_mat_lower.round(3), columns=dataset.name, index=dataset.name).to_csv('./assets/ACE_lower_{}.csv'.format(postfix))
    pd.DataFrame(ACE_mat_upper.round(3), columns=dataset.name, index=dataset.name).to_csv('./assets/ACE_upper_{}.csv'.format(postfix))
    #%%
    wandb.run.finish()
#%%
if __name__ == '__main__':
    main()
#%%
model_names = ['vanilla', 'InfoMax', 'gam']
lowers = {n : pd.read_csv('./assets/ACE_lower_{}.csv'.format(n), index_col=0) for n in model_names}
uppers = {n : pd.read_csv('./assets/ACE_upper_{}.csv'.format(n), index_col=0) for n in model_names}

fig, ax = plt.subplots(1, 4, figsize=(13, 3))
for i, s in enumerate(dataset.name):
    for n in model_names:
        ax[i].plot(lowers[n].loc[s], label=n)
    ax[i].set_ylabel('intervene: {}'.format(s))
    ax[i].set_ylim(0, 1)
    ax[i].legend()
plt.tight_layout()
# plt.savefig('./assets/ACE_metrics.png', bbox_inches='tight')
# # plt.show()
# plt.close()

# wandb.log({'ACE metrics (comparison)': wandb.Image(fig)})
#%%