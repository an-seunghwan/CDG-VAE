#%%
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.chdir(os.path.dirname(os.path.abspath(__file__)))
#%%
import numpy as np
import pandas as pd
import tqdm
from PIL import Image
import io
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
plt.switch_backend('agg')

from utils.simulation import (
    set_random_seed,
    is_dag,
    count_accuracy
)

from utils.viz import (
    viz_graph,
    viz_heatmap,
)

from utils.trac_exp import trace_expm
#%%
"""
wandb artifact cache cleanup "1GB"
"""
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
    tags=["causal_discovery"]
)
#%%
import argparse
def get_args(debug):
    parser = argparse.ArgumentParser('parameters')

    parser.add_argument('--seed', type=int, default=1, 
                        help='seed for repeatable results')
    
    parser.add_argument('--alpha', type=float, default=0.05, 
                        help='confidence level')

    if debug:
        return parser.parse_args(args=[])
    else:    
        return parser.parse_args()
#%%
def main():
    config = vars(get_args(debug=False)) # default configuration
    wandb.config.update(config)

    set_random_seed(config["seed"])

    train_imgs = [x for x in os.listdir('./utils/causal_data/pendulum/train') if x.endswith('png')]
    label = np.array([x[:-4].split('_')[1:] for x in train_imgs]).astype(float)
    label = label - label.mean(axis=0).round(2) 
    print("Label Dataset Mean:", label.mean(axis=0).round(2))
    print("Label Dataset Std:", label.std(axis=0).round(2))
    # label = label / label.std(axis=0)
    names = ['light', 'angle', 'length', 'position']
    
    data = label[np.random.choice(len(label), int(len(label) * 0.1), replace=False), :]
    # data = label

    from causallearn.utils.GraphUtils import GraphUtils
    
    """PC algorithm : CPDAG"""
    from causallearn.search.ConstraintBased.PC import pc
    cg = pc(data=data, 
            alpha=config["alpha"], 
            indep_test='kci') # Kernel-based conditional independence (KCI) test
    print(cg.G)
    # visualization
    pdy = GraphUtils.to_pydot(cg.G, labels=names)
    pdy.write_png('assets/graph_PC.png')
    fig = Image.open('assets/graph_PC.png')
    wandb.log({'PC_algorithm': wandb.Image(fig)})

    """FCI algorithm : PAG"""
    from causallearn.search.ConstraintBased.FCI import fci
    G, edges = fci(dataset=data, 
                   independence_test_method='kci', # Kernel-based conditional independence (KCI) test
                   alpha=config["alpha"])
    print(G) # https://www.nature.com/articles/s41598-020-59669-x
    print(G.graph)
    # visualization
    pdy = GraphUtils.to_pydot(G, labels=names)
    pdy.write_png('assets/graph_FCI.png')
    fig = Image.open('assets/graph_FCI.png')
    wandb.log({'FCI_algorithm': wandb.Image(fig)})
    
    # """GES algorithm"""
    # from causallearn.search.ScoreBased.GES import ges
    # Record = ges(X=data, 
    #              score_func='local_score_BIC')
    # # visualization
    # pdy = GraphUtils.to_pydot(Record['G'], labels=names)
    # pdy.write_png('assets/graph_GES.png')
    # fig = Image.open('assets/graph_GES.png')
    # wandb.log({'GES_algorithm': wandb.Image(fig)})
    
    # """Exact Search algorithm"""
    # from causallearn.search.ScoreBased.ExactSearch import bic_exact_search
    # dag_est, search_stats = bic_exact_search(X=data)
    # # visualization
    # fig = plt.figure(figsize=(5, 4))
    # plt.pcolor(np.flipud(dag_est), cmap='coolwarm')
    # plt.colorbar()
    # # plt.show()
    # plt.savefig('assets/graph_Exact.png')
    # plt.close()
    # wandb.log({'Exact_algorithm': wandb.Image(fig)})
    
    # """ICA-based LinGAM"""
    # from causallearn.search.FCMBased import lingam
    # model = lingam.ICALiNGAM()
    # model.fit(X=data)
    # # print(model.causal_order_)
    # print(model.adjacency_matrix_)
    # fig = plt.figure(figsize=(5, 4))
    # plt.pcolor(np.flipud(model.adjacency_matrix_), cmap='coolwarm')
    # plt.colorbar()
    # # plt.show()
    # plt.savefig('assets/graph_ICA_based_LinGAM.png')
    # plt.close()
    # wandb.log({'ICA_LinGAM_algorithm': wandb.Image(fig)})
    
    # """Direct LinGAM"""
    # from causallearn.search.FCMBased import lingam
    # model = lingam.DirectLiNGAM()
    # model.fit(X=data)
    # print(model.adjacency_matrix_)
    # fig = plt.figure(figsize=(5, 4))
    # plt.pcolor(np.flipud(model.adjacency_matrix_), cmap='coolwarm')
    # plt.colorbar()
    # # plt.show()
    # plt.savefig('assets/graph_Direct_LinGAM.png')
    # plt.close()
    # wandb.log({'Direct_LinGAM_algorithm': wandb.Image(fig)})
    
    wandb.run.finish()
#%%
if __name__ == '__main__':
    main()
#%%