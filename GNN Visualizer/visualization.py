import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from GraphUtility import edge_index_to_adjacency, normalize_adjacency, graph_to_networkx

# Part 6: Visualization Functions
def visualize_graph(data, title="Graph Visualization", node_colors=None, pos=None, figsize=(10, 8)):
    """Visualize a graph"""
    plt.figure(figsize=figsize)
    
    # Convert to NetworkX
    G = graph_to_networkx(data)
    
    # Get positions for nodes
    if pos is None:
        pos = nx.spring_layout(G, seed=42)
    
    # Set node colors
    if node_colors is None:
        if data.y is not None:
            node_colors = data.y.numpy()
        else:
            node_colors = 'lightblue'
    
    # Draw the graph
    nx.draw(G, pos, 
            node_color=node_colors,
            node_size=500,
            with_labels=True,
            font_size=12,
            font_weight='bold',
            edge_color='gray',
            cmap=plt.cm.Set3)
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.axis('off')
    return plt

def visualize_node_embeddings(embeddings, labels=None, title="Node Embeddings", figsize=(10, 8)):
    """Visualize node embeddings using t-SNE"""
    plt.figure(figsize=figsize)
    
    if embeddings.shape[1] > 2:
        # Use t-SNE for dimensionality reduction
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, embeddings.shape[0]-1))
        embeddings_2d = tsne.fit_transform(embeddings.detach().numpy())
    else:
        embeddings_2d = embeddings.detach().numpy()
    
    if labels is not None:
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                            c=labels.numpy(), cmap='viridis', s=100)
        plt.colorbar(scatter)
    else:
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], s=100)
    
    # Add node labels
    for i, (x, y) in enumerate(embeddings_2d):
        plt.annotate(f'N{i}', (x, y), xytext=(5, 5), 
                    textcoords='offset points', fontsize=12)
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    return plt

def visualize_adjacency_matrix(adj, title="Adjacency Matrix"):
    """Visualize adjacency matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(adj.numpy(), annot=True, cmap='Blues', 
                cbar_kws={'label': 'Connection'}, fmt='.1f')
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Node Index')
    plt.ylabel('Node Index')
    return plt