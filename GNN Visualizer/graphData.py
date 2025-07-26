import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import seaborn as sns


# Part 1: Graph Data Class
class GraphData:
    """Simple graph data structure"""
    def __init__(self, x, edge_index, y=None):
        self.x = x  # Node features [num_nodes, num_features]
        self.edge_index = edge_index  # Edges [2, num_edges]
        self.y = y  # Node labels [num_nodes]
        self.num_nodes = x.shape[0]
        self.num_edges = edge_index.shape[1]
        self.num_features = x.shape[1]

def create_sample_graph():
    """Create a sample graph for demonstration"""
    # Node features (5 nodes, 3 features each)
    x = torch.tensor([
        [1.0, 2.0, 3.0],  # Node 0
        [2.0, 1.0, 4.0],  # Node 1
        [3.0, 3.0, 1.0],  # Node 2
        [1.0, 4.0, 2.0],  # Node 3
        [4.0, 1.0, 1.0]   # Node 4
    ], dtype=torch.float)
    
    # Edge connections (undirected graph)
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 4, 0, 4],  # Source nodes
        [1, 0, 2, 1, 3, 2, 4, 3, 4, 0]   # Target nodes
    ], dtype=torch.long)
    
    # Node labels for classification
    y = torch.tensor([0, 1, 1, 0, 1], dtype=torch.long)
    
    return GraphData(x, edge_index, y)
