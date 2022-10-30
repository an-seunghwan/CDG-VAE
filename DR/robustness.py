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
from scipy.stats.contingency import crosstab

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
    tags=["VAEBased", "DistributionalRobustness"],
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
    model_name = 'GAM'
    # model_name = 'GAMsemi'
    
    scm = 'linear'
    # scm = 'nonlinear'
    
    """model load"""
    artifact = wandb.use_artifact('anseunghwan/CausalDisentangled/DRmodel_{}_{}:v{}'.format(model_name, scm, config["num"]), type='model')
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
    #%%
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
    #%%
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
        model.load_state_dict(torch.load(model_dir + '/DRmodel_{}_{}.pth'.format(config["model"], config["scm"])))
    else:
        model.load_state_dict(torch.load(model_dir + '/DRmodel_{}_{}.pth'.format(config["model"], config["scm"]), 
                                         map_location=torch.device('cpu')))
    
    model.eval()
    #%%
    """training dataset"""
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
    background = targets[:, [-2]]
    targets = targets[:, [-1]]
    representations = torch.cat(representations, dim=0)[:, :4]
    
    downstream_dataset = TensorDataset(representations, background, targets)
    downstream_dataloader = DataLoader(downstream_dataset, batch_size=64, shuffle=True)
    
    print('Train dataset label crosstab:')
    print(crosstab(background.cpu().numpy(), targets.cpu().numpy())[1] / len(targets))
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
    background = test_targets[:, [-2]]
    test_targets = test_targets[:, [-1]]
    test_representations = torch.cat(test_representations, dim=0)[:, :4]
    
    test_downstream_dataset = TensorDataset(test_representations, background, test_targets)
    test_downstream_dataloader = DataLoader(test_downstream_dataset, batch_size=64, shuffle=True)
    
    print('Test dataset label crosstab:')
    print(crosstab(background.cpu().numpy(), test_targets.cpu().numpy())[1] / len(test_targets))
    #%%
    accuracy_train = []
    worst_accuracy_train = []
    accuracy_test = []
    worst_accuracy_test = []
    for repeat_num in range(10): # repeated experiments
    
        downstream_classifier = DownstreamClassifier(config, device)
        downstream_classifier = downstream_classifier.to(device)
        
        optimizer = torch.optim.Adam(
            downstream_classifier.parameters(), 
            lr=0.005
        )
        
        downstream_classifier.train()
        
        for epoch in range(500):
            logs = {
                'loss': [], 
            }
            
            for (x_batch, background_batch, y_batch) in iter(downstream_dataloader):
                
                if config["cuda"]:
                    x_batch = x_batch.cuda()
                    background_batch = background_batch.cuda()
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
                worst_train_correct = 0
                worst_count = 0
                for (x_batch, background_batch, y_batch) in iter(downstream_dataloader):
                    
                    if config["cuda"]:
                        x_batch = x_batch.cuda()
                        background_batch = background_batch.cuda()
                        y_batch = y_batch.cuda()
                    
                    pred = downstream_classifier(x_batch)
                    pred = (pred > 0.5).float()
                    train_correct += (pred == y_batch).float().sum().item()
                    
                    opposite = torch.where((background_batch - y_batch) != 0)[0]
                    worst_train_correct += (pred[opposite] == y_batch[opposite]).float().sum().item()
                    worst_count += len(opposite)
                    
                train_correct /= downstream_dataset.__len__()
                worst_train_correct /= worst_count
                
                """test accuracy"""
                test_correct = 0
                worst_test_correct = 0
                worst_count = 0
                for (x_batch, background_batch, y_batch) in iter(test_downstream_dataloader):
                    
                    if config["cuda"]:
                        x_batch = x_batch.cuda()
                        background_batch = background_batch.cuda()
                        y_batch = y_batch.cuda()
                    
                    pred = downstream_classifier(x_batch)
                    pred = (pred > 0.5).float()
                    test_correct += (pred == y_batch).float().sum().item()
                    
                    opposite = torch.where((background_batch - y_batch) != 0)[0]
                    worst_test_correct += (pred[opposite] == y_batch[opposite]).float().sum().item()
                    worst_count += len(opposite)
                    
                test_correct /= test_downstream_dataset.__len__()
                worst_test_correct /= worst_count
                
            wandb.log({x : np.mean(y) for x, y in logs.items()})
            wandb.log({'AvgTrainACC(%)' : train_correct * 100})
            wandb.log({'AvgTestACC(%)' : test_correct * 100})
            wandb.log({'WorstTrainACC(%)' : worst_train_correct * 100})
            wandb.log({'WorstTestACC(%)' : worst_test_correct * 100})
        
        print_input = "[Repeat {:02d}]".format(repeat_num + 1)
        print_input += ''.join([', {}: {:.4f}'.format(x, np.mean(y)) for x, y in logs.items()])
        print_input += ', AvgTrainACC: {:.2f}%'.format(train_correct * 100)
        print_input += ', AvgTestACC: {:.2f}%'.format(test_correct * 100)
        print_input += ', WorstTrainACC: {:.2f}%'.format(worst_train_correct * 100)
        print_input += ', WorstTestACC: {:.2f}%'.format(worst_test_correct * 100)
        print(print_input)
        
        # log accuracy
        accuracy_train.append(train_correct)
        worst_accuracy_train.append(worst_train_correct)
        accuracy_test.append(test_correct)
        worst_accuracy_test.append(worst_test_correct)
    #%%
    """log Accuracy"""
    if not os.path.exists('./assets/robustness/'): 
        os.makedirs('./assets/robustness/')
    with open('./assets/robustness/{}_{}_{}.txt'.format(config["model"], config["scm"], config['num']), 'w') as f:
        f.write('train average accuracy: {:.4f}\n'.format(np.array(accuracy_train).mean()))
        f.write('train worst accuracy: {:.4f}\n'.format(np.array(worst_accuracy_train).mean()))
        f.write('test average accuracy: {:.4f}\n'.format(np.array(accuracy_test).mean()))
        f.write('test worst accuracy: {:.4f}\n'.format(np.array(worst_accuracy_test).mean()))
    #%%
    wandb.run.finish()
#%%
if __name__ == '__main__':
    main()
#%%
# model_names = ['VAE', 'InfoMax', 'CausalVAE', 'DEAR', 'GAM', 'GAMsemi']
# se_list = sorted(os.listdir('./assets/sample_efficiency/'))
# se_list = [x for x in se_list if x != '.DS_Store']

# se_result = {}
# for n in model_names:
#     file = [x for x in se_list if n in '_'.join(x.split('_')[:-1])]
#     if n == 'VAE':
#         file = file[1:]
#     for f in file:
#         with open('./assets/sample_efficiency/{}'.format(f)) as txt:
#             se_result['_'.join(f.split('_')[0:-1])] = [x.rstrip('\n') for x in txt.readlines()]

# with open('./assets/sample_efficiency/sample_efficiency.txt', 'w') as f:
#     f.write('100 samples accuracy, all samples accuracy, sample efficiency\n')
#     for key, value in se_result.items():
#         f.write('{})'.format(key.replace('_', '(')))
#         for x in [round(float(x.split(': ')[-1]) * 100, 2) for x in value]:
#             f.write(' & {:.2f}'.format(x))
#         # f.write(' & '.join([str(round(float(x.split(': ')[-1]) * 100, 2)) for x in value]))
#         f.write(' \\\\\n')
#%%