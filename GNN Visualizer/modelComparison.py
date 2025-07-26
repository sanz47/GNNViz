from training import train_gnn
from synthData import create_karate_club_graph
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from completeGNN import SimpleGCN, SimpleGAT
from synthData import create_karate_club_graph
# Part 9: Model Comparison
def compare_gnn_architectures():
    """Compare different GNN architectures"""
    print("Comparing GNN Architectures")
    print("-" * 30)
    
    # Create dataset
    data = create_karate_club_graph()
    print(f"Dataset: {data.num_nodes} nodes, {data.num_edges} edges, {data.num_features} features")
    
    # Initialize models
    models = {
        'GCN': SimpleGCN(input_dim=data.num_features, hidden_dim=16, output_dim=2),
        'GAT': SimpleGAT(input_dim=data.num_features, hidden_dim=16, output_dim=2),
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        losses, embeddings_history, adj_norm = train_gnn(model, data, epochs=200)
        
        # Final accuracy
        model.eval()
        with torch.no_grad():
            output = model(data.x, adj_norm)
            pred = output.argmax(dim=1)
            accuracy = (pred == data.y).float().mean().item()
        
        results[name] = {
            'losses': losses,
            'embeddings': embeddings_history,
            'accuracy': accuracy,
            'final_embeddings': embeddings_history[-1] if embeddings_history else None
        }
        print(f"{name} final accuracy: {accuracy:.3f}")
    
    # Visualize results
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('GNN Architecture Comparison', fontsize=16)
    
    # Plot losses
    for i, (name, result) in enumerate(results.items()):
        axes[0, i].plot(result['losses'])
        axes[0, i].set_title(f'{name} Training Loss (Acc: {result["accuracy"]:.3f})')
        axes[0, i].set_xlabel('Epoch')
        axes[0, i].set_ylabel('Loss')
        axes[0, i].grid(True)
    
    # Plot final embeddings
    for i, (name, result) in enumerate(results.items()):
        if result['final_embeddings'] is not None:
            embeddings = result['final_embeddings']
            if embeddings.shape[1] > 2:
                perplexity = min(30, embeddings.shape[0] - 1, max(1, embeddings.shape[0] // 3))
                tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
                emb_2d = tsne.fit_transform(embeddings.detach().numpy())
            else:
                emb_2d = embeddings.detach().numpy()
            
            scatter = axes[1, i].scatter(emb_2d[:, 0], emb_2d[:, 1], 
                                       c=data.y.numpy(), cmap='viridis', s=50)
            axes[1, i].set_title(f'{name} Final Embeddings')
            
            # Add colorbar
            plt.colorbar(scatter, ax=axes[1, i])
    
    plt.tight_layout()
    return plt, results