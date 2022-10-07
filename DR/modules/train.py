#%%
import tqdm

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import Dataset
#%%
def train_VAE(dataloader, model, config, optimizer, device):
    logs = {
        'loss': [], 
        'recon': [],
        'KL': [],
        'alignment': [],
    }
    # for debugging
    for i in range(config["node"]):
        logs['posterior_variance{}'.format(i+1)] = []
    
    for (x_batch, y_batch) in tqdm.tqdm(iter(dataloader), desc="inner loop"):
        
        if config["cuda"]:
            x_batch = x_batch.cuda()
            y_batch = y_batch.cuda()
        
        # with torch.autograd.set_detect_anomaly(True):    
        optimizer.zero_grad()
        
        mean, logvar, _, _, _, _, align_latent, xhat = model(x_batch)
        
        loss_ = []
        
        """reconstruction"""
        recon = 0.5 * torch.pow(xhat - x_batch, 2).sum(axis=[1, 2, 3]).mean() 
        # recon = F.mse_loss(xhat, x_batch)
        loss_.append(('recon', recon))
        
        """KL-Divergence"""
        KL = torch.pow(mean, 2).sum(axis=1)
        KL -= logvar.sum(axis=1)
        KL += torch.exp(logvar).sum(axis=1)
        KL -= config["node"]
        KL *= 0.5
        KL = KL.mean()
        loss_.append(('KL', KL))
        
        """Label Alignment : CrossEntropy"""
        y_hat = torch.sigmoid(torch.cat(align_latent, dim=1))
        align = F.binary_cross_entropy(y_hat, y_batch[:, :config["node"]], reduction='none').sum(axis=1).mean()
        loss_.append(('alignment', align))
        
        ### posterior variance: for debugging
        var_ = torch.exp(logvar).mean(axis=0)
        for i in range(config["node"]):
            loss_.append(('posterior_variance{}'.format(i+1), var_[i]))
        
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
def train_InfoMax(dataloader, model, discriminator, config, optimizer, optimizer_D, device):
    
    def permute_dims(z, device):
        B, _ = z.size()
        perm = torch.randperm(B).to(device)
        perm_z = z[perm]
        return perm_z

    logs = {
        'loss': [], 
        'recon': [],
        'KL': [],
        'alignment': [],
        'MutualInfo': [],
    }
    # for debugging
    for i in range(config["node"]):
        logs['posterior_variance{}'.format(i+1)] = []
    
    for (x_batch, y_batch) in tqdm.tqdm(iter(dataloader), desc="inner loop"):
        
        if config["cuda"]:
            x_batch = x_batch.cuda()
            y_batch = y_batch.cuda()
        
        # with torch.autograd.set_detect_anomaly(True):    
        mean, logvar, epsilon, _, _, _, align_latent, xhat = model(x_batch)
        
        loss_ = []
        
        """reconstruction"""
        recon = 0.5 * torch.pow(xhat - x_batch, 2).sum(axis=[1, 2, 3]).mean() 
        # recon = F.mse_loss(xhat, x_batch)
        loss_.append(('recon', recon))
        
        """KL-Divergence"""
        KL = torch.pow(mean, 2).sum(axis=1)
        KL -= logvar.sum(axis=1)
        KL += torch.exp(logvar).sum(axis=1)
        KL -= config["node"]
        KL *= 0.5
        KL = KL.mean()
        loss_.append(('KL', KL))
        
        """Label Alignment : CrossEntropy"""
        y_hat = torch.sigmoid(torch.cat(align_latent, dim=1))
        align = F.binary_cross_entropy(y_hat, y_batch[:, :config["node"]], reduction='none').sum(axis=1).mean()
        loss_.append(('alignment', align))
        
        """mutual information"""
        D_joint = discriminator(x_batch, epsilon)
        epsilon_perm = permute_dims(epsilon, device)
        D_marginal = discriminator(x_batch, epsilon_perm)
        MI = -(D_joint.mean() - torch.exp(D_marginal - 1).mean())
        loss_.append(('MutualInfo', MI))
        
        ### posterior variance: for debugging
        var_ = torch.exp(logvar).mean(axis=0)
        for i in range(config["node"]):
            loss_.append(('posterior_variance{}'.format(i+1), var_[i]))
        
        loss = recon + config["beta"] * KL 
        loss += config["lambda"] * align
        loss += config["gamma"] * MI
        loss_.append(('loss', loss))
        
        optimizer.zero_grad()
        optimizer_D.zero_grad()
        loss.backward(retain_graph=True)
        MI.backward()
        optimizer.step()
        optimizer_D.step()
            
        """accumulate losses"""
        for x, y in loss_:
            logs[x] = logs.get(x) + [y.item()]
    
    return logs, xhat
#%%
def train_GAM(dataloader, model, config, optimizer, device):
    logs = {
        'loss': [], 
        'recon': [],
        'KL': [],
        'alignment': [],
    }
    # for debugging
    for i in range(config["node"]):
        logs['posterior_variance{}'.format(i+1)] = []
    
    for (x_batch, y_batch) in tqdm.tqdm(iter(dataloader), desc="inner loop"):
        
        if config["cuda"]:
            x_batch = x_batch.cuda()
            y_batch = y_batch.cuda()
        
        # with torch.autograd.set_detect_anomaly(True):    
        optimizer.zero_grad()
        
        mean, logvar, _, _, _, _, align_latent, _, xhat = model(x_batch)
        
        loss_ = []
        
        """reconstruction"""
        recon = 0.5 * torch.pow(xhat - x_batch, 2).sum(axis=[1, 2, 3]).mean() 
        # recon = F.mse_loss(xhat, x_batch)
        loss_.append(('recon', recon))
        
        """KL-Divergence"""
        KL = torch.pow(mean, 2).sum(axis=1)
        KL -= logvar.sum(axis=1)
        KL += torch.exp(logvar).sum(axis=1)
        KL -= config["node"]
        KL *= 0.5
        KL = KL.mean()
        loss_.append(('KL', KL))
        
        """Label Alignment : CrossEntropy"""
        y_hat = torch.sigmoid(torch.cat(align_latent, dim=1))
        align = F.binary_cross_entropy(y_hat, y_batch[:, :config["node"]], reduction='none').sum(axis=1).mean()
        loss_.append(('alignment', align))
        
        ### posterior variance: for debugging
        var_ = torch.exp(logvar).mean(axis=0)
        for i in range(config["node"]):
            loss_.append(('posterior_variance{}'.format(i+1), var_[i]))
        
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
def train_GAM_semi(datasetL, datasetU, model, config, optimizer, device):
    logs = {
        'loss': [], 
        'recon': [],
        'KL': [],
        'alignment': [],
    }
    # for debugging
    for i in range(config["node"]):
        logs['posterior_variance{}'.format(i+1)] = []
    
    dataloaderU = DataLoader(datasetU, batch_size=config["batch_size"], shuffle=True)
    dataloaderL = DataLoader(datasetL, batch_size=config["batch_sizeL"], shuffle=True)
        
    for x_batchU in tqdm.tqdm(iter(dataloaderU), desc="inner loop"):
        try:
            x_batchL, y_batchL = next(iter_dataloaderL)
        except:
            iter_dataloaderL = iter(dataloaderL)
            x_batchL, y_batchL = next(iter_dataloaderL)
        
        if config["cuda"]:
            x_batchU = x_batchU.cuda()
            x_batchL = x_batchL.cuda()
            y_batchL = y_batchL.cuda()
        
        # with torch.autograd.set_detect_anomaly(True):    
        optimizer.zero_grad()
        
        # unsupervised
        mean, logvar, _, _, _, _, align_latent, _, xhat = model(x_batchU)
        
        loss_ = []
        
        """reconstruction"""
        recon = 0.5 * torch.pow(xhat - x_batchU, 2).sum(axis=[1, 2, 3]).mean() 
        # recon = F.mse_loss(xhat, x_batch)
        loss_.append(('recon', recon))
        
        """KL-Divergence"""
        KL = torch.pow(mean, 2).sum(axis=1)
        KL -= logvar.sum(axis=1)
        KL += torch.exp(logvar).sum(axis=1)
        KL -= config["node"]
        KL *= 0.5
        KL = KL.mean()
        loss_.append(('KL', KL))
        
        # supervised
        """Label Alignment : CrossEntropy"""
        _, _, _, _, align_latent, _ = model.encode(x_batchL, deterministic=True)
        y_hat = torch.sigmoid(torch.cat(align_latent, dim=1))
        align = F.binary_cross_entropy(y_hat, y_batchL[:, :config["node"]], reduction='none').sum(axis=1).mean()
        loss_.append(('alignment', align))
        
        ### posterior variance: for debugging
        var_ = torch.exp(logvar).mean(axis=0)
        for i in range(config["node"]):
            loss_.append(('posterior_variance{}'.format(i+1), var_[i]))
        
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