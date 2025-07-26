import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from graphAttention import GraphAttention

from GraphConvolutional import GraphConvolution
# Part 5: Complete GNN Models
class SimpleGCN(nn.Module):
    """Simple Graph Convolutional Network"""
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super(SimpleGCN, self).__init__()
        self.gc1 = GraphConvolution(input_dim, hidden_dim)
        self.gc2 = GraphConvolution(hidden_dim, output_dim)
        self.dropout = dropout
        
    def forward(self, x, adj):
        # First layer
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        
        # Second layer
        x = self.gc2(x, adj)
        
        return x

class SimpleGAT(nn.Module):
    """Simple Graph Attention Network"""
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super(SimpleGAT, self).__init__()
        self.gat1 = GraphAttention(input_dim, hidden_dim, dropout)
        self.gat2 = GraphAttention(hidden_dim, output_dim, dropout)
        self.dropout = dropout
        
    def forward(self, x, adj):
        # First layer
        x = F.relu(self.gat1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        
        # Second layer
        x = self.gat2(x, adj)
        
        return x