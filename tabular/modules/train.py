#%%
import tqdm

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import Dataset
from torch.nn.functional import cross_entropy
#%%
def train_VAE(dataset, dataloader, model, config, optimizer, device):
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
        if config["dataset"] == 'loan':
            recon = 0.5 * torch.pow(xhat - x_batch[:, dataset.flatten_topology], 2).sum(axis=1).mean() 
        elif config["dataset"] == 'adult':
            x_batch_ = x_batch[:, dataset.flatten_topology]
            recon = 0.5 * torch.pow(xhat[:, :2] - x_batch_[:, :2], 2).sum(axis=1).mean() 
            recon += 0.5 * torch.pow(xhat[:, 3:] - x_batch_[:, 3:], 2).sum(axis=1).mean() 
            recon += F.binary_cross_entropy_with_logits(xhat[:, [2]], x_batch_[:, [2]]) # income
        else:
            raise ValueError('Not supported dataset!')
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
        align = F.binary_cross_entropy(y_hat, y_batch, reduction='none').sum(axis=1).mean()
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
    
    return logs
#%%
def train_InfoMax(dataset, dataloader, model, discriminator, config, optimizer, optimizer_D, device):
    
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
        if config["dataset"] == 'loan':
            recon = 0.5 * torch.pow(xhat - x_batch[:, dataset.flatten_topology], 2).sum(axis=1).mean() 
        elif config["dataset"] == 'adult':
            x_batch_ = x_batch[:, dataset.flatten_topology]
            recon = 0.5 * torch.pow(xhat[:, :2] - x_batch_[:, :2], 2).sum(axis=1).mean() 
            recon += 0.5 * torch.pow(xhat[:, 3:] - x_batch_[:, 3:], 2).sum(axis=1).mean() 
            recon += F.binary_cross_entropy_with_logits(xhat[:, [2]], x_batch_[:, [2]]) # income
        else:
            raise ValueError('Not supported dataset!')
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
        align = F.binary_cross_entropy(y_hat, y_batch, reduction='none').sum(axis=1).mean()
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
    
    return logs
#%%
def train_GAM(dataset, dataloader, model, config, optimizer, device):
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
        if config["dataset"] == 'loan':
            recon = 0.5 * torch.pow(xhat - x_batch[:, dataset.flatten_topology], 2).sum(axis=1).mean() 
        elif config["dataset"] == 'adult':
            x_batch_ = x_batch[:, dataset.flatten_topology]
            recon = 0.5 * torch.pow(xhat[:, :2] - x_batch_[:, :2], 2).sum(axis=1).mean() 
            recon += 0.5 * torch.pow(xhat[:, 3:] - x_batch_[:, 3:], 2).sum(axis=1).mean() 
            recon += F.binary_cross_entropy_with_logits(xhat[:, [2]], x_batch_[:, [2]]) # income
        else:
            raise ValueError('Not supported dataset!')
        loss_.append(('recon', recon))
        
        """KL-Divergence"""
        KL = torch.pow(mean, 2).sum(axis=1)
        KL -= logvar.sum(axis=1)
        KL += torch.exp(logvar).sum(axis=1)
        KL -= config["node"]
        KL *= 0.5
        KL = KL.mean()
        loss_.append(('KL', KL))
        
        """Label Alignment"""
        y_hat = torch.sigmoid(torch.cat(align_latent, dim=1))
        align = F.binary_cross_entropy(y_hat, y_batch, reduction='none').sum(axis=1).mean()
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
    
    return logs
#%%
def train_TVAE(output_info_list, dataset, dataloader, model, config, optimizer, device):
    logs = {
        'loss': [], 
        'recon': [],
        'KL': [],
        'alignment': []
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
        start = 0
        recon = 0
        for column_info in output_info_list:
            for span_info in column_info:
                if span_info.activation_fn != 'softmax':
                    end = start + span_info.dim
                    std = model.sigma[start]
                    residual = x_batch[:, start] - torch.tanh(xhat[:, start])
                    recon += (residual ** 2 / 2 / (std ** 2)).mean()
                    recon += torch.log(std)
                    start = end
                else:
                    end = start + span_info.dim
                    recon += cross_entropy(
                        xhat[:, start:end], torch.argmax(x_batch[:, start:end], dim=-1), reduction='mean')
                    start = end
        loss_.append(('recon', recon))
        
        """KL-Divergence"""
        KL = torch.pow(mean, 2).sum(axis=1)
        KL -= logvar.sum(axis=1)
        KL += torch.exp(logvar).sum(axis=1)
        KL -= config["node"]
        KL *= 0.5
        KL = KL.mean()
        loss_.append(('KL', KL))
        
        """Label Alignment"""
        y_hat = torch.sigmoid(torch.cat(align_latent, dim=1))
        align = F.binary_cross_entropy(y_hat, y_batch, reduction='none').sum(axis=1).mean()
        loss_.append(('alignment', align))
        
        ### posterior variance: for debugging
        var_ = torch.exp(logvar).mean(axis=0)
        for i in range(config["node"]):
            loss_.append(('posterior_variance{}'.format(i+1), var_[i]))
        
        loss = recon + KL 
        loss += config["lambda"] * align
        loss_.append(('loss', loss))
        
        loss.backward()
        optimizer.step()
        model.sigma.data.clamp_(0.01, 0.1)
            
        """accumulate losses"""
        for x, y in loss_:
            logs[x] = logs.get(x) + [y.item()]
    
    return logs
#%%