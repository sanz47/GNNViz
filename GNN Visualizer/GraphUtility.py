import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Part 2: Graph Utility Functions
def edge_index_to_adjacency(edge_index, num_nodes):
    """Convert edge index to adjacency matrix"""
    adj = torch.zeros(num_nodes, num_nodes)
    adj[edge_index[0], edge_index[1]] = 1.0
    return adj

def normalize_adjacency(adj):
    """Normalize adjacency matrix (D^-0.5 * A * D^-0.5)"""
    # Add self-loops
    adj = adj + torch.eye(adj.size(0))
    
    # Compute degree matrix
    degree = torch.sum(adj, dim=1)
    degree_inv_sqrt = torch.pow(degree, -0.5)
    degree_inv_sqrt[degree_inv_sqrt == float('inf')] = 0.0
    
    # Create normalized adjacency matrix
    degree_matrix_inv_sqrt = torch.diag(degree_inv_sqrt)
    adj_normalized = torch.mm(torch.mm(degree_matrix_inv_sqrt, adj), degree_matrix_inv_sqrt)
    
    return adj_normalized

def graph_to_networkx(data):
    """Convert our graph data to NetworkX graph"""
    G = nx.Graph()
    G.add_nodes_from(range(data.num_nodes))
    
    # Add edges
    edge_list = data.edge_index.t().numpy()
    edges = [(int(edge[0]), int(edge[1])) for edge in edge_list]
    G.add_edges_from(edges)
    
    return G
