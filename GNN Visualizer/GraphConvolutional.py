import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Part 3: Graph Convolutional Layer (from scratch)
class GraphConvolution(nn.Module):
    """Graph Convolution Layer implementation from scratch"""
    def __init__(self, input_dim, output_dim, bias=True):
        super(GraphConvolution, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Learnable parameters
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(output_dim))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters"""
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x, adj):
        """Forward pass"""
        # Linear transformation: X * W
        support = torch.mm(x, self.weight)
        
        # Graph convolution: A * X * W
        output = torch.mm(adj, support)
        
        if self.bias is not None:
            output = output + self.bias
            
        return output