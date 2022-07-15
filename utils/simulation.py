#%%
import igraph as ig
import networkx as nx
import numpy as np
import random
#%%
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
#%%
def is_dag(W: np.ndarray):
    """check DAGness
    Args:
        W: weight matrix of graph
    Returns:
        Boolean: True if W is weight matrix of DAG
    """
    G = ig.Graph.Weighted_Adjacency(W.tolist())
    return G.is_dag()
#%%