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
    TabularDataset, 
    TestTabularDataset,
)
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
    
    scm = 'linear'
    # scm = 'nonlinear'
    
    """model load"""
    artifact = wandb.use_artifact(
        'anseunghwan/CausalDisentangled/tabular_model_{}_{}:v{}'.format(
            model_name, scm, config["num"]), type='model')
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
    dataset = TabularDataset(config)
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)
    
    from causallearn.search.ConstraintBased.PC import pc
    from causallearn.utils.GraphUtils import GraphUtils
    
    cg = pc(data=dataset.train.to_numpy(), 
            alpha=0.05, 
            indep_test='chisq') 
    print(cg.G)
    trainG = cg.G.graph
    
    # visualization
    pdy = GraphUtils.to_pydot(cg.G, labels=dataset.continuous)
    pdy.write_png('./assets/loan/dag_train_loan.png')
    fig = Image.open('./assets/loan/dag_train_loan.png')
    wandb.log({'Baseline DAG (Train)': wandb.Image(fig)})
    #%%
    """
    Causal Adjacency Matrix
    [Mortgage, Income] -> CCAvg
    [Experience, Age] -> CCAvg
    """
    B = torch.zeros(config["node"], config["node"])
    B[:-1, -1] = 1
    
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
        mask = [2, 2, 1]
        from modules.model import GAM
        model = GAM(B, mask, config, device) 
    
    else:
        raise ValueError('Not supported model!')
        
    model = model.to(device)
    
    if config["cuda"]:
        model.load_state_dict(
            torch.load(
                model_dir + '/tabular_model_{}_{}.pth'.format(
                    config["model"], config["scm"])))
    else:
        model.load_state_dict(
            torch.load(
                model_dir + '/tabular_model_{}_{}.pth'.format(
                    config["model"], config["scm"]), 
                        map_location=torch.device('cpu')))
    
    model.eval()
    #%%
    df = pd.read_csv('./data/Bank_Personal_Loan_Modelling.csv')
    df = df.sample(frac=1, random_state=1).reset_index(drop=True)
    df = df.drop(columns=['ID'])
    continuous = ['CCAvg', 'Mortgage', 'Income', 'Experience', 'Age']
    df = df[continuous]
    
    df_ = (df - df.mean(axis=0)) / df.std(axis=0)
    train = df_.iloc[:4000]
    test = df_.iloc[4000:]
    
    cg = pc(data=train.to_numpy(), 
            alpha=0.05, 
            indep_test='chisq') 
    print(cg.G)
    trainG = cg.G.graph
    
    # visualization
    pdy = GraphUtils.to_pydot(cg.G, labels=df.columns)
    pdy.write_png('./assets/loan/dag_train_loan.png')
    fig = Image.open('./assets/loan/dag_train_loan.png')
    # wandb.log({'Baseline DAG (Train)': wandb.Image(fig)})
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
    train_recon = dataset.transformer.inverse_transform(train_recon, model.sigma.detach().cpu().numpy())
    #%%
    """PC algorithm : train dataset representation"""
    train_recon = (train_recon - train_recon.mean(axis=0)) / train_recon.std(axis=0)
    train_recon = train_recon[['CCAvg', 'Mortgage', 'Income', 'Experience', 'Age']]
    cg = pc(data=train_recon.to_numpy(), 
            alpha=0.05, 
            indep_test='fisherz') 
    print(cg.G)
    
    # SHD: https://arxiv.org/pdf/1306.1043.pdf
    trainSHD = (np.triu(trainG) != np.triu(cg.G.graph)).sum() # unmatch in upper-triangular
    nonzero_idx = np.where(np.triu(cg.G.graph) != 0)
    flag = np.triu(trainG)[nonzero_idx] == np.triu(cg.G.graph)[nonzero_idx]
    nonzero_idx = (nonzero_idx[1][flag], nonzero_idx[0][flag])
    trainSHD += (np.tril(trainG)[nonzero_idx] != np.tril(cg.G.graph)[nonzero_idx]).sum()
    # wandb.log({'SHD (Train)': trainSHD})
    
    # visualization
    pdy = GraphUtils.to_pydot(cg.G, labels=train_recon.columns)
    pdy.write_png('./assets/loan/dag_recon_train_loan.png')
    fig = Image.open('./assets/loan/dag_recon_train_loan.png')
    # wandb.log({'Reconstructed DAG (Train)': wandb.Image(fig)})
    #%%
    """synthetic dataset"""
    torch.manual_seed(config["seed"])
    steps = len(train) // config["batch_size"] + 1
    data = []
    with torch.no_grad():
        for _ in range(steps):
            mean = torch.zeros(config["batch_size"], config["node"])
            std = mean + 1
            noise = torch.normal(mean=mean, std=std).to(device)
            _, latent, _ = model.transform(noise, log_determinant=False)
            fake = model.decode(latent)[1]
            fake = torch.tanh(fake)
            data.append(fake.numpy())
    data = np.concatenate(data, axis=0)
    data = data[:len(train)]
    sample_df = dataset.transformer.inverse_transform(data, model.sigma.detach().cpu().numpy())
    #%%
    """PC algorithm : synthetic dataset"""
    sample_df = (sample_df - sample_df.mean(axis=0)) / sample_df.std(axis=0)
    sample_df = sample_df[['CCAvg', 'Mortgage', 'Income', 'Experience', 'Age']]
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
    # wandb.log({'SHD (Sample)': sampleSHD})
    
    # visualization
    pdy = GraphUtils.to_pydot(cg.G, labels=sample_df.columns)
    pdy.write_png('./assets/loan/dag_recon_sample_loan.png')
    fig = Image.open('./assets/loan/dag_recon_sample_loan.png')
    # wandb.log({'Reconstructed DAG (Sampled)': wandb.Image(fig)})
    #%%
    """Machine Learning Efficacy"""
    import statsmodels.api as sm
    
    # Baseline
    covariates = [x for x in train.columns if x != 'CCAvg']
    linreg = sm.OLS(train['CCAvg'], train[covariates]).fit()
    linreg.summary()
    pred = linreg.predict(test[covariates])
    rsq_baseline = 1 - (test['CCAvg'] - pred).pow(2).sum() / np.var(test['CCAvg']) / len(test)
    print("Baseline R-squared: {:.2f}".format(rsq_baseline))
    # wandb.log({'R^2 (Baseline)': rsq_baseline})
    #%%
    # Train
    covariates = [x for x in sample_df.columns if x != 'CCAvg']
    linreg = sm.OLS(sample_df['CCAvg'], sample_df[covariates]).fit()
    linreg.summary()
    pred = linreg.predict(test[covariates])
    rsq = 1 - (test['CCAvg'] - pred).pow(2).sum() / np.var(test['CCAvg']) / len(test)
    print("CDG-TVAE R-squared: {:.2f}".format(rsq))
    # wandb.log({'R^2 (Sample)': rsq})
    #%%
    wandb.run.finish()
#%%
if __name__ == '__main__':
    main()
#%%