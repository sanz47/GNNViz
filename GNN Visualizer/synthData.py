import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from graphData import create_sample_graph, GraphData
# Part 8: Create Larger Synthetic Dataset
def create_karate_club_graph():
    """Create Karate Club graph dataset"""
    # Use NetworkX Karate Club graph
    G = nx.karate_club_graph()
    
    # Convert to our format
    edge_list = list(G.edges())
    edge_index = torch.tensor([[u, v] for u, v in edge_list] + 
                             [[v, u] for u, v in edge_list], dtype=torch.long).t()
    
    # Create node features based on degree and clustering
    num_nodes = G.number_of_nodes()
    features = []
    
    for i in range(num_nodes):
        degree = G.degree(i)
        clustering = nx.clustering(G, i)
        centrality = nx.degree_centrality(G)[i]
        features.append([degree, clustering, centrality])
    
    x = torch.tensor(features, dtype=torch.float)
    
    # Create labels based on community (Mr. Hi vs Officer)
    y = torch.tensor([0 if G.nodes[i]['club'] == 'Mr. Hi' else 1 
                      for i in range(num_nodes)], dtype=torch.long)
    
    return GraphData(x, edge_index, y)

def create_synthetic_graph(num_nodes=20, num_features=5):
    """Create a synthetic graph for experimentation"""
    # Generate random features
    x = torch.randn(num_nodes, num_features)
    
    # Create a connected graph with community structure
    G = nx.planted_partition_graph(2, num_nodes//2, 0.7, 0.1, seed=42)
    
    # Convert to edge index
    edge_list = list(G.edges())
    edge_index = torch.tensor([[u, v] for u, v in edge_list] + 
                             [[v, u] for u, v in edge_list], dtype=torch.long).t()
    
    # Create labels based on community
    y = torch.tensor([0 if i < num_nodes//2 else 1 for i in range(num_nodes)], dtype=torch.long)
    
    return GraphData(x, edge_index, y)