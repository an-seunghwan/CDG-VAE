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

import statsmodels.api as sm
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
#%%
import sys
import subprocess
try:
    import wandb
except:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "wandb"])
    with open("./wandb_api.txt", "r") as f:
        key = f.readlines()
    subprocess.run(["wandb", "login"], input=key[0], encoding='utf-8')
    import wandb

run = wandb.init(
    project="CausalDisentangled", 
    entity="anseunghwan",
    tags=["Tabular", "VAEBased", "Inference"],
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
    
    dataset = 'loan'
    # config["dataset"] = 'loan'
    # dataset = 'adult'
    # dataset = 'covtype'
    
    """model load"""
    artifact = wandb.use_artifact(
        'anseunghwan/CausalDisentangled/tabular_{}_{}:v{}'.format(
            model_name, dataset, config["num"]), type='model')
    for key, item in artifact.metadata.items():
        config[key] = item
    assert model_name == config["model"]
    assert dataset == config["dataset"]
    model_dir = artifact.download()
    
    config["cuda"] = torch.cuda.is_available()
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    wandb.config.update(config)
    
    set_random_seed(config["seed"])
    torch.manual_seed(config["seed"])
    if config["cuda"]:
        torch.cuda.manual_seed(config["seed"])
    #%%
    import importlib
    dataset_module = importlib.import_module('modules.{}_datasets'.format(config["dataset"]))
    TabularDataset = dataset_module.TabularDataset
    TestTabularDataset = dataset_module.TestTabularDataset
    
    dataset = TabularDataset(config)
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)
    #%%
    """
    Causal Adjacency Matrix
    Personal Loan:
        [Mortgage, Income] -> CCAvg
        [Experience, Age] -> CCAvg
    Adult:
        capital-gain -> [income, educational-num, hours-per-week]
        capital-loss -> [income, educational-num, hours-per-week]
    Forest Cover Type Prediction:
    """
    if config["dataset"] == 'loan':
        B = torch.zeros(config["node"], config["node"])
        B[:-1, -1] = 1
        
        i_test = 'chisq'
    
    elif config["dataset"] == 'adult':
        B = torch.zeros(config["node"], config["node"])
        B[:-1, -1] = 1
        
        i_test = 'chisq'
    
    elif config["dataset"] == 'covtype':
        B = torch.zeros(config["node"], config["node"])
        B[[0, 3, 4, 5], 1] = 1
        B[[3, 4, 5], 2] = 1
        B[[0, 5], 3] = 1
        
        i_test = 'fisherz'
        
    else:
        raise ValueError('Not supported dataset!')
    
    """adjacency matrix scaling"""
    if config["adjacency_scaling"]:
        indegree = B.sum(axis=0)
        mask = (indegree != 0)
        B[:, mask] = B[:, mask] / indegree[mask]
    print(B)
    #%%
    """model"""
    if config["model"] == 'VAE':
        from modules.model import VAE
        model = VAE(B, config, device) 
        
    elif config["model"] == 'InfoMax':
        from modules.model import VAE, Discriminator
        model = VAE(B, config, device) 
        discriminator = Discriminator(config, device)
        discriminator = discriminator.to(device)
        
        optimizer_D = torch.optim.Adam(
            discriminator.parameters(), 
            lr=config["lr_D"]
        )
        
    elif config["model"] == 'GAM':
        """Decoder masking"""
        if config["dataset"] == 'loan':
            mask = [2, 2, 1]
        elif config["dataset"] == 'adult':
            mask = [1, 1, 3]
        elif config["dataset"] == 'covtype':
            mask = [1, 1, 2, 1, 1, 1 + 7]
        else:
            raise ValueError('Not supported dataset!')
        from modules.model import GAM
        model = GAM(B, mask, config, device) 
    
    else:
        raise ValueError('Not supported model!')
        
    model = model.to(device)
    
    if config["cuda"]:
        model_name = [x for x in os.listdir(model_dir) if x.endswith('pth')][0]
        model.load_state_dict(
            torch.load(
                model_dir + '/' + model_name))
    else:
        model_name = [x for x in os.listdir(model_dir) if x.endswith('pth')][0]
        model.load_state_dict(
            torch.load(
                model_dir + '/' + model_name, map_location=torch.device('cpu')))
    
    model.eval()
    #%%
    from causallearn.search.ConstraintBased.PC import pc
    from causallearn.utils.GraphUtils import GraphUtils
    
    cg = pc(data=dataset.train.to_numpy(), 
            alpha=0.05, 
            indep_test=i_test) 
    print(cg.G)
    trainG = cg.G.graph
    
    # visualization
    pdy = GraphUtils.to_pydot(cg.G, labels=dataset.continuous)
    pdy.write_png('./assets/{}/dag_train_{}.png'.format(config["dataset"], config["dataset"]))
    fig = Image.open('./assets/{}/dag_train_{}.png'.format(config["dataset"], config["dataset"]))
    wandb.log({'Baseline DAG (Train)': wandb.Image(fig)})
    #%%
    testdataset = TestTabularDataset(config)
    testdataloader = DataLoader(testdataset, batch_size=config["batch_size"], shuffle=True)
    #%%
    """train dataset representation"""
    train_recon = []
    for (x_batch, y_batch) in tqdm.tqdm(iter(dataloader), desc="inner loop"):
        if config["cuda"]:
            x_batch = x_batch.cuda()
            y_batch = y_batch.cuda()
        
        with torch.no_grad():
            out = model(x_batch, deterministic=True)
        train_recon.append(out[-1])
    train_recon = torch.cat(train_recon, dim=0)
    if config["dataset"] == "covtype":
        train_recon = torch.cat([
            train_recon[:, :7],
            train_recon[:, 7:].argmax(axis=1, keepdim=True)], dim=1)
    #%%
    """synthetic dataset"""
    torch.manual_seed(config["seed"])
    randn = torch.randn(dataset.__len__(), config["node"])
    with torch.no_grad():
        _, latent, _ = model.transform(randn, log_determinant=False)
        if config["model"] == 'GAM':
            sample_recon = model.decode(latent)[1]
        else:
            sample_recon = model.decoder(torch.cat(latent, dim=1))
    if config["dataset"] == "covtype":
        sample_recon = torch.cat([
            sample_recon[:, :7],
            sample_recon[:, 7:].argmax(axis=1, keepdim=True)], dim=1)
    #%%
    """PC algorithm : train dataset representation"""
    cols = [item for sublist in dataset.topology for item in sublist]
    train_df = pd.DataFrame(train_recon.numpy(), columns=cols)
    train_df = train_df[dataset.continuous]
    if 'income' in dataset.continuous:
        train_df['income'] = train_df['income'].apply(lambda x: 1 if x > 0 else 0)
    
    cg = pc(data=train_df.to_numpy(), 
            alpha=0.05, 
            indep_test='fisherz') 
    print(cg.G)
    
    # SHD: https://arxiv.org/pdf/1306.1043.pdf
    trainSHD = (np.triu(trainG) != np.triu(cg.G.graph)).sum() # unmatch in upper-triangular
    nonzero_idx = np.where(np.triu(cg.G.graph) != 0)
    flag = np.triu(trainG)[nonzero_idx] == np.triu(cg.G.graph)[nonzero_idx]
    nonzero_idx = (nonzero_idx[1][flag], nonzero_idx[0][flag])
    trainSHD += (np.tril(trainG)[nonzero_idx] != np.tril(cg.G.graph)[nonzero_idx]).sum()
    print('SHD (Train): {}'.format(trainSHD))
    wandb.log({'SHD (Train)': trainSHD})
    
    # visualization
    pdy = GraphUtils.to_pydot(cg.G, labels=train_df.columns)
    pdy.write_png('./assets/{}/dag_recon_train_{}.png'.format(config["dataset"], config["dataset"]))
    fig = Image.open('./assets/{}/dag_recon_train_{}.png'.format(config["dataset"], config["dataset"]))
    wandb.log({'Reconstructed DAG (Train)': wandb.Image(fig)})
    #%%
    """PC algorithm : synthetic dataset"""
    cols = [item for sublist in dataset.topology for item in sublist]
    sample_df = pd.DataFrame(sample_recon.numpy(), columns=cols)
    sample_df = sample_df[dataset.continuous]
    if 'income' in dataset.continuous:
        sample_df['income'] = sample_df['income'].apply(lambda x: 1 if x > 0 else 0)
    
    cg = pc(data=sample_df.to_numpy(), 
            alpha=0.05, 
            indep_test='fisherz') 
    print(cg.G)
    
    # SHD: https://arxiv.org/pdf/1306.1043.pdf
    sampleSHD = (np.triu(trainG) != np.triu(cg.G.graph)).sum() # unmatch in upper-triangular
    nonzero_idx = np.where(np.triu(cg.G.graph) != 0)
    flag = np.triu(trainG)[nonzero_idx] == np.triu(cg.G.graph)[nonzero_idx]
    nonzero_idx = (nonzero_idx[1][flag], nonzero_idx[0][flag])
    sampleSHD += (np.tril(trainG)[nonzero_idx] != np.tril(cg.G.graph)[nonzero_idx]).sum()
    print('SHD (Sample): {}'.format(sampleSHD))
    wandb.log({'SHD (Sample)': sampleSHD})
    
    # visualization
    pdy = GraphUtils.to_pydot(cg.G, labels=sample_df.columns)
    pdy.write_png('./assets/{}/dag_recon_sample_{}.png'.format(config["dataset"], config["dataset"]))
    fig = Image.open('./assets/{}/dag_recon_sample_{}.png'.format(config["dataset"], config["dataset"]))
    wandb.log({'Reconstructed DAG (Sampled)': wandb.Image(fig)})
    #%%
    """Machine Learning Efficacy"""
    
    # Baseline
    if config["dataset"] == 'loan':
        covariates = [x for x in dataset.train.columns if x != 'CCAvg']
        
        linreg = sm.OLS(dataset.train['CCAvg'], dataset.train[covariates]).fit()
        pred = linreg.predict(testdataset.test[covariates])
        rsq_baseline = 1 - (testdataset.test['CCAvg'] - pred).pow(2).sum() / np.var(testdataset.test['CCAvg']) / testdataset.__len__()
        
        print("Baseline R-squared: {:.2f}".format(rsq_baseline))
        wandb.log({'R^2 (Baseline)': rsq_baseline})
        
    elif config["dataset"] == 'adult':
        covariates = [x for x in dataset.train.columns if x != 'income']
        
        clf = RandomForestClassifier(random_state=0)
        clf.fit(dataset.train[covariates], dataset.train['income'])
        pred = clf.predict(testdataset.test[covariates])
        pred = (pred > 0.5).astype(float)
        f1_baseline = f1_score(testdataset.test['income'], pred)
        
        print("Baseline F1: {:.2f}".format(f1_baseline))
        wandb.log({'F1 (Baseline)': f1_baseline})
    
    elif config["dataset"] == 'covtype':
        covariates = [x for x in dataset.train.columns if x != 'Cover_Type']
        
        clf = RandomForestClassifier(random_state=0)
        clf.fit(dataset.train[covariates], dataset.train['Cover_Type'])
        pred = clf.predict(testdataset.test[covariates])
        f1_baseline = f1_score(testdataset.test['Cover_Type'].to_numpy(), pred, average='micro')
        
        print("Baseline F1: {:.2f}".format(f1_baseline))
        wandb.log({'F1 (Baseline)': f1_baseline})
    
    else:
        raise ValueError('Not supported dataset!')
    #%%
    # synthetic
    if config["dataset"] == 'loan':
        covariates = [x for x in sample_df.columns if x != 'CCAvg']
        linreg = sm.OLS(sample_df['CCAvg'], sample_df[covariates]).fit()
        linreg.summary()
        pred = linreg.predict(testdataset.test[covariates])
        rsq = 1 - (testdataset.test['CCAvg'] - pred).pow(2).sum() / np.var(testdataset.test['CCAvg']) / testdataset.__len__()
        print("{}-{} R-squared: {:.2f}".format(config["model"], config["dataset"], rsq))
        wandb.log({'R^2 (Sample)': rsq})
        
    elif config["dataset"] == 'adult':
        if 'income' in dataset.continuous:
            sample_df['income'] = sample_df['income'].apply(lambda x: 1 if x > 0 else 0)
        covariates = [x for x in dataset.train.columns if x != 'income']
        
        clf = RandomForestClassifier(random_state=0)
        clf.fit(sample_df[covariates], sample_df['income'])
        pred = clf.predict(testdataset.test[covariates])
        pred = (pred > 0.5).astype(float)
        f1 = f1_score(testdataset.test['income'], pred)
        
        print("{}-{} F1: {:.2f}".format(config["model"], config["dataset"], f1))
        wandb.log({'F1 (Sample)': f1})
    
    elif config["dataset"] == 'covtype':
        covariates = [x for x in dataset.train.columns if x != 'Cover_Type']
        
        clf = RandomForestClassifier(random_state=0)
        clf.fit(sample_df[covariates], sample_df['Cover_Type'])
        pred = clf.predict(testdataset.test[covariates])
        f1 = f1_score(testdataset.test['Cover_Type'].to_numpy(), pred, average='micro')
        
        print("{}-{} F1: {:.2f}".format(config["model"], config["dataset"], f1))
        wandb.log({'F1 (Sample)': f1})
    
    else:
        raise ValueError('Not supported dataset!')
    #%%
    wandb.run.finish()
#%%
if __name__ == '__main__':
    main()
#%%