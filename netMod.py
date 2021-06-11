import math
from community import community_louvain
import networkx as nx
import numpy as np
import pandas as pd


##########################
# versione della correlation network che usa networkx
def createCorrelationNetwork(corr_, min_correlation=0.3):
    corr = pd.DataFrame(corr_)
    links = corr.stack().reset_index()
    links.columns = ['var1', 'var2', 'weight']
    # Keep only correlation over a threshold and remove self correlation (cor(A,A)=1)
    links_filtered = links.loc[(links['weight'] > min_correlation) & (links['var1'] != links['var2'])]
    # Crates graph using the data of the correlation matrix
    G = nx.from_pandas_edgelist(links_filtered, 'var1', 'var2', edge_attr=True)
    return G


##################################################
"""
Use of Louvain method to indetify communities
"""
def getNetPartition(corr):
    # init net
    g = createCorrelationNetwork(corr)  # networkx graph
    partition = community_louvain.best_partition(g)
    # take communities
    list_nodes = []
    for com in set(partition.values()):  # creates arrays with nodes in a cluster
        ln = [nodes for nodes in partition.keys()
              if partition[nodes] == com]
        list_nodes.append(ln)
    for i in range(corr.shape[0]):  # add those that are not in a cluster
        if i not in partition.keys():
            list_nodes.append([i])
    return list_nodes

##################################################
"""
Idea: group based on correlation matrix.
"""
def getNetMod(corr):
    partition = getNetPartition(corr)
    w = np.zeros(corr.shape[0])
    weight_per_cluster = 1 / len(partition)
    for cluster in partition:
        for i in cluster:
            w[i] = weight_per_cluster / len(cluster)
    return w