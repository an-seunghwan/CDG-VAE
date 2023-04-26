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

from modules.model import TVAE

from modules.evaluation import (
    regression_eval,
    classification_eval,
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
    tags=["Tabular", "VAEBased", "Inference2"],
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
    
    model_name = 'TVAE'
    
    # dataset = 'loan'
    # dataset = 'adult'
    dataset = 'covtype'
    
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
    TabularDataset2 = dataset_module.TabularDataset2
    
    if config["dataset"] == 'loan':
        dataset = TabularDataset2(config, random_state=8)
    elif config["dataset"] == 'adult':
        dataset = TabularDataset2(config) # 0
    elif config["dataset"] == 'covtype':
        dataset = TabularDataset2(config, random_state=0)
    else:
        raise ValueError('Not supported dataset!')
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)
    
    config["input_dim"] = dataset.transformer.output_dimensions
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
    
    elif config["dataset"] == 'adult':
        B = torch.zeros(config["node"], config["node"])
        B[:-1, -1] = 1
    
    elif config["dataset"] == 'covtype':
        B = torch.zeros(config["node"], config["node"])
        B[[0, 3, 4, 5], 1] = 1
        B[[3, 4, 5], 2] = 1
        B[[0, 5], 3] = 1
        
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
    decoder_dims = []
    for l in dataset.transformer.output_info_list:
        decoder_dims.append(sum([x.dim for x in l]))
    
    if config["dataset"] == 'loan':
        mask_ = [0, 2, 2, 1]
        mask_ = np.cumsum(mask_)
    elif config["dataset"] == 'adult':
        mask_ = [0, 1, 1, 3]
        mask_ = np.cumsum(mask_)
    elif config["dataset"] == 'covtype':
        mask_ = [0, 1, 1, 2, 1, 1, 1 + 7]
        mask_ = np.cumsum(mask_)
    else:
        raise ValueError('Not supported dataset!')
    
    mask = []
    for j in range(len(mask_) - 1):
        mask.append(sum(decoder_dims[mask_[j]:mask_[j+1]]))
    
    model = TVAE(B, mask, config, device).to(device)
    
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
    
    if config["dataset"] == 'loan':
        df = pd.read_csv('./data/Bank_Personal_Loan_Modelling.csv')
        df = df.sample(frac=1, random_state=1).reset_index(drop=True)
        df = df.drop(columns=['ID'])
        continuous = ['CCAvg', 'Mortgage', 'Income', 'Experience', 'Age']
        df = df[continuous]
        
        df_ = (df - df.mean(axis=0)) / df.std(axis=0)
        train = df_.iloc[:4000]
        test = df_.iloc[4000:]
        
        i_test = 'chisq'
        
    elif config["dataset"] == 'adult':
        df = pd.read_csv('./data/adult.csv')
        df = df.sample(frac=1, random_state=1).reset_index(drop=True)
        df = df[(df == '?').sum(axis=1) == 0]
        df['income'] = df['income'].map({'<=50K': 0, '>50K': 1, '<=50K.': 0, '>50K.': 1})
        df = df[dataset.continuous]
        
        # scaling = [x for x in dataset.continuous if x != 'income']
        # df_ = df.copy()
        # df_[scaling] = (df[scaling] - df[scaling].mean(axis=0)) / df[scaling].std(axis=0)
        train = df.iloc[:40000]
        test = df.iloc[40000:]
        
        i_test = 'chisq'
        
    elif config["dataset"] == 'covtype':
        df = pd.read_csv('./data/covtype.csv')
        df = df.sample(frac=1, random_state=5).reset_index(drop=True)
        df = df[dataset.continuous]
        df = df.dropna(axis=0)
        
        train = df.iloc[2000:, ]
        test = df.iloc[:2000, ]
        
        i_test = 'fisherz'
        
    else:
        raise ValueError('Not supported dataset!')
    
    cg = pc(data=train.to_numpy(), 
            alpha=0.05, 
            indep_test=i_test) 
    print(cg.G)
    trainG = cg.G.graph
    
    # visualization
    pdy = GraphUtils.to_pydot(cg.G, labels=df.columns)
    pdy.write_png('./assets/{}/dag_train_{}.png'.format(config["dataset"], config["dataset"]))
    fig = Image.open('./assets/{}/dag_train_{}.png'.format(config["dataset"], config["dataset"]))
    wandb.log({'Baseline DAG (Train)': wandb.Image(fig)})
    #%%
    def gumbel_sampling(size, eps = 1e-20):
        U = torch.rand(size)
        G = (- (U + eps).log() + eps).log()
        return G
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
    #%%
    if config["dataset"] == "covtype":
        train_recon1 = dataset.transformer.inverse_transform(train_recon.numpy(), model.sigma.detach().cpu().numpy())
        out = train_recon[:, -7:]
        G = gumbel_sampling(out.shape)
        _, out = (nn.LogSoftmax(dim=1)(out) + G).max(dim=1, keepdims=True)
        train_recon1["Cover_Type"] = out
        train_recon = train_recon1
    else:
        train_recon = dataset.transformer.inverse_transform(train_recon.numpy(), model.sigma.detach().cpu().numpy())
    #%%
    """PC algorithm : train dataset representation"""
    try: 
        train_df = pd.DataFrame(train_recon.to_numpy(), columns=dataset.flatten_topology)
    except:
        cols = [item for sublist in dataset.topology for item in sublist]
        train_df = pd.DataFrame(train_recon.to_numpy(), columns=cols)
    train_df = train_df[df.columns]
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
    pdy = GraphUtils.to_pydot(cg.G, labels=train_recon.columns)
    pdy.write_png('./assets/{}/dag_recon_train_{}.png'.format(config["dataset"], config["dataset"]))
    fig = Image.open('./assets/{}/dag_recon_train_{}.png'.format(config["dataset"], config["dataset"]))
    wandb.log({'Reconstructed DAG (Train)': wandb.Image(fig)})
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
    #%%
    if config["dataset"] == "covtype":
        sample_df1 = dataset.transformer.inverse_transform(data, model.sigma.detach().cpu().numpy())
        out = torch.tensor(data[:, -7:])
        G = gumbel_sampling(out.shape)
        _, out = (nn.LogSoftmax(dim=1)(out) + G).max(dim=1, keepdims=True)
        sample_df1["Cover_Type"] = out
        sample_df = sample_df1
    else:
        sample_df = dataset.transformer.inverse_transform(data, model.sigma.detach().cpu().numpy())
    #%%
    """PC algorithm : synthetic dataset"""
    try:
        sample_df = pd.DataFrame(sample_df.to_numpy(), columns=dataset.flatten_topology)
    except:
        cols = [item for sublist in dataset.topology for item in sublist]
        sample_df = pd.DataFrame(sample_df.to_numpy(), columns=cols)
    sample_df = sample_df[df.columns]
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
    if config["dataset"] == "loan": # regression
        target = 'CCAvg'
        
        # baseline
        print("\nBaseline: Machine Learning Utility in Regression...\n")
        base_r2result = regression_eval(train, test, target)
        wandb.log({'R^2 (Baseline)': np.mean([x[1] for x in base_r2result])})
        
        # Synthetic
        print("\nSynthetic: Machine Learning Utility in Regression...\n")
        r2result = regression_eval(sample_df, test, target)
        wandb.log({'R^2 (Synthetic)': np.mean([x[1] for x in r2result])})
    
    elif config["dataset"] == "adult": # classification
        target = 'income' 
        
        # baseline
        print("\nBaseline: Machine Learning Utility in Classification...\n")
        base_f1result = classification_eval(train, test, target)
        wandb.log({'F1 (Baseline)': np.mean([x[1] for x in base_f1result])})
        
        # Synthetic
        print("\nSynthetic: Machine Learning Utility in Classification...\n")
        f1result = classification_eval(sample_df, test, target)
        wandb.log({'F1 (Synthetic)': np.mean([x[1] for x in f1result])})
        
    elif config["dataset"] == "covtype": # classification
        target = 'Cover_Type'
        
        # # baseline
        # print("\nBaseline: Machine Learning Utility in Classification...\n")
        # base_f1result = classification_eval(train, test, target)
        # wandb.log({'F1 (Baseline)': np.mean([x[1] for x in base_f1result])})
        
        # Synthetic
        print("\nSynthetic: Machine Learning Utility in Classification...\n")
        f1result = classification_eval(sample_df, test, target)
        wandb.log({'F1 (Synthetic)': np.mean([x[1] for x in f1result])})
        
    else:
        raise ValueError('Not supported dataset!')
    #%%
    wandb.run.finish()
#%%
if __name__ == '__main__':
    main()
#%%