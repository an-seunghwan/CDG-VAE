#%%
import tqdm

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import Dataset
#%%
def train_CDGVAE(train_loader, model, config, optimizer, device):
    logs = {
        'loss': [], 
        'recon': [],
        'KL': [],
        'alignment': [],
        'active': [],
    }
    
    for (x_batch, y_batch) in tqdm.tqdm(iter(train_loader), desc="inner loop"):
        
        if config["cuda"]:
            x_batch = x_batch.cuda()
            y_batch = y_batch.cuda()
        
        # with torch.autograd.set_detect_anomaly(True):    
        optimizer.zero_grad()
        
        (mean1, logvar1, epsilon1, orig_latent, latent, logdet), (mean2, logvar2, epsilon2), align_latent, xhat_separated, xhat = model(x_batch)
        
        loss_ = []
        
        """reconstruction"""
        x_batch_ = x_batch[..., :3] * 2 - 1
        recon = (xhat - x_batch_).abs().sum(axis=[1, 2, 3]).mean() 
        loss_.append(('recon', recon))
        
        """KL-Divergence"""
        KL1 = torch.pow(mean1, 2).sum(axis=1)
        KL1 -= logvar1.sum(axis=1)
        KL1 += torch.exp(logvar1).sum(axis=1)
        KL1 -= config["node"]
        KL1 *= 0.5
        KL1 = KL1.mean()
        
        KL2 = torch.pow(mean2, 2).sum(axis=1)
        KL2 -= logvar2.sum(axis=1)
        KL2 += torch.exp(logvar2).sum(axis=1)
        KL2 -= config["node"]
        KL2 *= 0.5
        KL2 = KL2.mean()
        
        loss_.append(('KL', KL1 + KL2))
        
        """Label Alignment : CrossEntropy"""
        y_hat = torch.sigmoid(torch.cat(align_latent, dim=1))
        align = F.binary_cross_entropy(y_hat, y_batch[:, :config["node"]], reduction='none').sum(axis=1).mean()
        loss_.append(('alignment', align))
        
        ### posterior variance: for debugging
        active = (torch.exp(logvar1).mean(axis=0) < 0.1).to(torch.float32).sum()
        active += (torch.exp(logvar2).mean(axis=0) < 0.1).to(torch.float32).sum()
        active /= config["node"] + config["latent_dim"]
        loss_.append(('active', active))
        
        loss = recon + config["beta"] * (KL1 + KL2) 
        loss += config["lambda"] * align
        loss_.append(('loss', loss))
        
        loss.backward()
        optimizer.step()
            
        """accumulate losses"""
        for x, y in loss_:
            logs[x] = logs.get(x) + [y.item()]
    
    return logs, xhat
#%%