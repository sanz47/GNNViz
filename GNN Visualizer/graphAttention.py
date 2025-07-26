import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Part 4: Graph Attention Layer (simplified)
class GraphAttention(nn.Module):
    """Simplified Graph Attention Layer"""
    def __init__(self, input_dim, output_dim, dropout=0.1, alpha=0.2):
        super(GraphAttention, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.alpha = alpha
        
        # Linear transformation
        self.W = nn.Parameter(torch.empty(size=(input_dim, output_dim)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        
        # Attention mechanism
        self.a = nn.Parameter(torch.empty(size=(2 * output_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        
        self.leakyrelu = nn.LeakyReLU(self.alpha)
    
    def forward(self, x, adj):
        """Forward pass"""
        # Linear transformation
        Wh = torch.mm(x, self.W)  # [N, output_dim]
        N = Wh.size()[0]
        
        # Compute attention coefficients
        # Create all pairs for attention computation
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)
        
        # Concatenate features
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        
        # Compute attention scores
        e = self.leakyrelu(torch.mm(all_combinations_matrix, self.a)).squeeze(1)
        e = e.view(N, N)
        
        # Mask attention for non-adjacent nodes
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        
        # Apply softmax
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        # Apply attention to features
        h_prime = torch.mm(attention, Wh)
        
        return h_prime