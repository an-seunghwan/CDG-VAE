#%%
import igraph as ig
import networkx as nx
import numpy as np
import random
import torch
#%%
"""for reproducibility"""
def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    
    # random.seed(seed)
    # np.random.seed(seed)
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
def count_accuracy(B_true, B_est):
    """Compute various accuracy metrics for B_est.
    reverse = predicted association exists in condition in opposite direction
    false positive = predicted association does not exist in condition
    
    Note that: Partial DAG is not considered!
    
    Args:
        B_true (np.ndarray): d x d ground truth binary adjacency graph, {0, 1}
        B_est (np.ndarray): d x d estimated binary adjacency graph, {0, 1}
        
    Returns:
        FDR: (reverse + false positive) / prediction 
        SHD: (undirected) extra + (undirected) missing + reverse
        nonzero: prediction positive
    """
    
    if not ((B_est == 0) | (B_est == 1)).all():
        raise ValueError('B_est should take value in {0,1}')
    if not is_dag(B_est):
        raise ValueError('B_est should be a DAG')

    d = B_true.shape[0]

    # linear index of nonzero weights
    pred = np.flatnonzero(B_est == 1)
    cond = np.flatnonzero(B_true)
    cond_reversed = np.flatnonzero(B_true.T)
    cond_skeleton = np.concatenate([cond, cond_reversed])

    # false pos
    false_pos = np.setdiff1d(pred, cond_skeleton, assume_unique=True)
    # reverse
    extra = np.setdiff1d(pred, cond, assume_unique=True)
    reverse = np.intersect1d(extra, cond_reversed, assume_unique=True)
    # compute ratio
    FDR = float(len(false_pos) + len(reverse)) / max(len(pred), 1)

    # structural hamming distance
    pred_lower = np.flatnonzero(np.tril(B_est + B_est.T))
    cond_lower = np.flatnonzero(np.tril(B_true + B_true.T))
    extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
    missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True)
    SHD = len(extra_lower) + len(missing_lower) + len(reverse)

    return {'FDR': FDR, 'SHD': SHD, 'nonzero': len(pred)}
#%%