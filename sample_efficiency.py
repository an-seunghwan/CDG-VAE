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

from modules.datasets import (
    LabeledDataset, 
    UnLabeledDataset,
    TestDataset,
)

from modules.model import (
    DownstreamClassifier
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
    tags=["VAEBased", "SampleEfficiency"],
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
    dataset = LabeledDataset(config, downstream=True)
    test_dataset = TestDataset(config, downstream=True)

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
        
    elif config["model"] in ['CDGVAE', 'CDGVAEsemi']:
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
        
        from modules.model import CDGVAE
        model = CDGVAE(B, mask, config, device) 
    
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
    beta = torch.tensor([[1, -1, 0.5, -0.5]]).to(device)
    
    """with 100 size of training dataset"""
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    targets_100 = []
    representations_100 = []
    for count, (x_batch, y_batch) in tqdm.tqdm(enumerate(iter(dataloader))):
        if config["cuda"]:
            x_batch = x_batch.cuda()
            y_batch = y_batch.cuda()
        
        with torch.no_grad():
            mean, logvar = model.get_posterior(x_batch)
        targets_100.append(y_batch)
        representations_100.append(mean)
        
        count += 1
        if count == 100: break
    targets_100 = torch.cat(targets_100, dim=0)
    logit = torch.matmul(targets_100[:, :-1], beta.t())
    targets_100 = torch.bernoulli(1 / (1 + torch.exp(-logit - 2*torch.sin(logit))))
    representations_100 = torch.cat(representations_100, dim=0)
    
    downstream_dataset_100 = TensorDataset(representations_100, targets_100)
    downstream_dataloader_100 = DataLoader(downstream_dataset_100, batch_size=32, shuffle=True)
    #%%
    """with all training dataset"""
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    targets = []
    representations = []
    for x_batch, y_batch in tqdm.tqdm(iter(dataloader)):
        if config["cuda"]:
            x_batch = x_batch.cuda()
            y_batch = y_batch.cuda()
        
        with torch.no_grad():
            mean, logvar = model.get_posterior(x_batch)
        targets.append(y_batch)
        representations.append(mean)
        
    targets = torch.cat(targets, dim=0)
    logit = torch.matmul(targets[:, :-1], beta.t())
    targets = torch.bernoulli(1 / (1 + torch.exp(-logit - 2*torch.sin(logit))))
    representations = torch.cat(representations, dim=0)
    
    downstream_dataset = TensorDataset(representations, targets)
    downstream_dataloader = DataLoader(downstream_dataset, batch_size=64, shuffle=True)
    #%%
    """test dataset"""
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)
    test_targets = []
    test_representations = []
    for x_batch, y_batch in tqdm.tqdm(iter(test_dataloader)):
        if config["cuda"]:
            x_batch = x_batch.cuda()
            y_batch = y_batch.cuda()
        
        with torch.no_grad():
            mean, logvar = model.get_posterior(x_batch)
        test_targets.append(y_batch)
        test_representations.append(mean)
        
    test_targets = torch.cat(test_targets, dim=0)
    logit = torch.matmul(test_targets[:, :-1], beta.t())
    test_targets = torch.bernoulli(1 / (1 + torch.exp(-logit - 2*torch.sin(logit))))
    test_representations = torch.cat(test_representations, dim=0)
    
    test_downstream_dataset = TensorDataset(test_representations, test_targets)
    test_downstream_dataloader = DataLoader(test_downstream_dataset, batch_size=64, shuffle=True)
    #%%
    accuracy = []
    accuracy_100 = []
    for repeat_num in range(10): # repeated experiments
    
        print("Sample Efficiency with 100 labels")
        downstream_classifier_100 = DownstreamClassifier(config, device)
        downstream_classifier_100 = downstream_classifier_100.to(device)
        
        optimizer = torch.optim.Adam(
            downstream_classifier_100.parameters(), 
            lr=0.005
        )
        
        downstream_classifier_100.train()
        
        for epoch in range(100):
            logs = {
                'loss': [], 
            }
            
            for (x_batch, y_batch) in iter(downstream_dataloader_100):
                
                if config["cuda"]:
                    x_batch = x_batch.cuda()
                    y_batch = y_batch.cuda()
                
                # with torch.autograd.set_detect_anomaly(True):    
                optimizer.zero_grad()
                
                pred = downstream_classifier_100(x_batch)
                loss = F.binary_cross_entropy(pred, y_batch, reduction='none').mean()
                
                loss_ = []
                loss_.append(('loss', loss))
                
                loss.backward()
                optimizer.step()
                    
                """accumulate losses"""
                for x, y in loss_:
                    logs[x] = logs.get(x) + [y.item()]
            
            # accuracy
            with torch.no_grad():
                """train accuracy"""
                train_correct = 0
                for (x_batch, y_batch) in iter(downstream_dataloader_100):
                    
                    if config["cuda"]:
                        x_batch = x_batch.cuda()
                        y_batch = y_batch.cuda()
                    
                    pred = downstream_classifier_100(x_batch)
                    pred = (pred > 0.5).float()
                    train_correct += (pred == y_batch).float().sum().item()
                train_correct /= downstream_dataset_100.__len__()
                
                """test accuracy"""
                test_correct = 0
                for (x_batch, y_batch) in iter(test_downstream_dataloader):
                    
                    if config["cuda"]:
                        x_batch = x_batch.cuda()
                        y_batch = y_batch.cuda()
                    
                    pred = downstream_classifier_100(x_batch)
                    pred = (pred > 0.5).float()
                    test_correct += (pred == y_batch).float().sum().item()
                test_correct /= test_downstream_dataset.__len__()
            
            wandb.log({x : np.mean(y) for x, y in logs.items()})
            wandb.log({'TrainACC(%)_100samples' : train_correct * 100})
            wandb.log({'TestACC(%)_100samples' : test_correct * 100})
        
        print_input = "[Repeat {:02d}]".format(repeat_num + 1)
        print_input += ''.join([', {}: {:.4f}'.format(x, np.mean(y).round(2)) for x, y in logs.items()])
        print_input += ', TrainACC: {:.2f}%'.format(train_correct * 100)
        print_input += ', TestACC: {:.2f}%'.format(test_correct * 100)
        print(print_input)
            
        # log accuracy
        accuracy_100.append(test_correct)
        
        print("Sample Efficiency with all labels")
        downstream_classifier = DownstreamClassifier(config, device)
        downstream_classifier = downstream_classifier.to(device)
        
        optimizer = torch.optim.Adam(
            downstream_classifier.parameters(), 
            lr=0.005
        )
        
        downstream_classifier.train()
        
        for epoch in range(100):
            logs = {
                'loss': [], 
            }
            
            for (x_batch, y_batch) in iter(downstream_dataloader):
                
                if config["cuda"]:
                    x_batch = x_batch.cuda()
                    y_batch = y_batch.cuda()
                
                # with torch.autograd.set_detect_anomaly(True):    
                optimizer.zero_grad()
                
                pred = downstream_classifier(x_batch)
                loss = F.binary_cross_entropy(pred, y_batch, reduction='none').mean()
                
                loss_ = []
                loss_.append(('loss', loss))
                
                loss.backward()
                optimizer.step()
                    
                """accumulate losses"""
                for x, y in loss_:
                    logs[x] = logs.get(x) + [y.item()]
            
            # accuracy
            with torch.no_grad():
                """train accuracy"""
                train_correct = 0
                for (x_batch, y_batch) in iter(downstream_dataloader):
                    
                    if config["cuda"]:
                        x_batch = x_batch.cuda()
                        y_batch = y_batch.cuda()
                    
                    pred = downstream_classifier(x_batch)
                    pred = (pred > 0.5).float()
                    train_correct += (pred == y_batch).float().sum().item()
                train_correct /= downstream_dataset.__len__()
                
                """test accuracy"""
                test_correct = 0
                for (x_batch, y_batch) in iter(test_downstream_dataloader):
                    
                    if config["cuda"]:
                        x_batch = x_batch.cuda()
                        y_batch = y_batch.cuda()
                    
                    pred = downstream_classifier(x_batch)
                    pred = (pred > 0.5).float()
                    test_correct += (pred == y_batch).float().sum().item()
                test_correct /= test_downstream_dataset.__len__()
            
            wandb.log({x : np.mean(y) for x, y in logs.items()})
            wandb.log({'TrainACC(%)' : train_correct * 100})
            wandb.log({'TestACC(%)' : test_correct * 100})
        
        print_input = "[Repeat {:02d}]".format(repeat_num + 1)
        print_input += ''.join([', {}: {:.4f}'.format(x, np.mean(y).round(2)) for x, y in logs.items()])
        print_input += ', TrainACC: {:.2f}%'.format(train_correct * 100)
        print_input += ', TestACC: {:.2f}%'.format(test_correct * 100)
        print(print_input)
        
        # log accuracy
        accuracy.append(test_correct)
    #%%
    """log Accuracy"""
    sample_efficiency = np.array(accuracy_100).mean() / np.array(accuracy).mean()
    if not os.path.exists('./assets/sample_efficiency/'): 
        os.makedirs('./assets/sample_efficiency/')
    with open('./assets/sample_efficiency/{}_{}_{}.txt'.format(config["model"], config["scm"], config['num']), 'w') as f:
        f.write('100 samples accuracy: {:.4f}\n'.format(np.array(accuracy_100).mean()))
        f.write('all samples accuracy: {:.4f}\n'.format(np.array(accuracy).mean()))
        f.write('sample efficiency: {:.4f}\n'.format(sample_efficiency))
    #%%
    wandb.run.finish()
#%%
if __name__ == '__main__':
    main()
#%%