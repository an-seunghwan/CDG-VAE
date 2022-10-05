#%%
import matplotlib.pyplot as plt

import networkx as nx
#%%
def viz_graph(W, size=(6, 6), show=False):
    """visualize weighted adj matrix of DAG"""
    fig = plt.figure(figsize=size)
    G = nx.from_numpy_matrix(W, create_using=nx.DiGraph)
    try:
        layout = nx.planar_layout(G)
    except:
        layout = nx.spectral_layout(G)
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw(G, layout, 
            with_labels=True, 
            font_size=20,
            font_weight='bold',
            arrowsize=40,
            node_size=1000)
    nx.draw_networkx_edge_labels(G, 
                                pos=layout, 
                                edge_labels=labels, 
                                font_weight='bold',
                                font_size=15)
    if show:
        plt.show()
    plt.close()
    return fig
#%%
def viz_heatmap(W, size=(5, 4), show=False):
    """visualize heatmap of weighted adj matrix of DAG"""
    fig = plt.figure(figsize=size)
    plt.pcolor(W, cmap='coolwarm')
    plt.colorbar()
    if show:
        plt.show()
    plt.close()
    return fig
#%%