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
# Part 7: Training Functions
def train_gnn(model, data, epochs=100, lr=0.01):
    """Train a GNN model"""
    # Convert edge index to adjacency matrix
    adj = edge_index_to_adjacency(data.edge_index, data.num_nodes)
    adj_norm = normalize_adjacency(adj)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    losses = []
    embeddings_history = []
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward pass
        output = model(data.x, adj_norm)
        loss = criterion(output, data.y)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        # Store embeddings every 20 epochs for visualization
        if epoch % 20 == 0:
            with torch.no_grad():
                model.eval()
                embeddings = model(data.x, adj_norm)
                embeddings_history.append(embeddings.clone())
                model.train()
    
    return losses, embeddings_history, adj_norm

def plot_training_progress(losses, embeddings_history, labels, model_name="GNN"):
    """Plot training loss and embedding evolution"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'{model_name} Training Progress', fontsize=16)
    
    # Plot training loss
    axes[0, 0].plot(losses)
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True)
    
    # Plot embedding evolution
    for i, embeddings in enumerate(embeddings_history[:5]):  # Show first 5 checkpoints
        row = i // 3
        col = (i + 1) % 3
        
        if embeddings.shape[1] > 2:
            # Handle small datasets for t-SNE
            perplexity = min(30, embeddings.shape[0] - 1, max(1, embeddings.shape[0] // 3))
            tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
            emb_2d = tsne.fit_transform(embeddings.detach().numpy())
        else:
            emb_2d = embeddings.detach().numpy()
        
        scatter = axes[row, col].scatter(emb_2d[:, 0], emb_2d[:, 1], 
                                       c=labels.numpy(), cmap='viridis', s=100)
        axes[row, col].set_title(f'Epoch {i * 20}')
        
        # Add node labels
        for j, (x, y) in enumerate(emb_2d):
            axes[row, col].annotate(f'N{j}', (x, y), xytext=(3, 3), 
                                  textcoords='offset points', fontsize=10)
    
    plt.tight_layout()
    return plt