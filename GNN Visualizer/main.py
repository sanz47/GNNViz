import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from modelComparison import compare_gnn_architectures
from training import train_gnn, plot_training_progress
from visualization import visualize_graph
# from graphAttention
from completeGNN import SimpleGCN
from GraphConvolutional import GraphConvolution
from graphData import create_sample_graph, GraphData
from GraphUtility import edge_index_to_adjacency, normalize_adjacency, graph_to_networkx
from synthData  import create_karate_club_graph
# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

print("=== Graph Neural Networks Tutorial (Pure PyTorch) ===\n")

def main_demo():
    """Main demonstration function"""
    print("=== Running GNN Tutorial Demo ===\n")
    
    # 1. Create and visualize sample graph
    print("1. Creating and visualizing sample graph...")
    sample_data = create_sample_graph()
    print(f"Sample graph: {sample_data.num_nodes} nodes, {sample_data.num_edges} edges")
    
    adj = edge_index_to_adjacency(sample_data.edge_index, sample_data.num_nodes)
    adj_norm = normalize_adjacency(adj)
    
    # Visualizations
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    G = graph_to_networkx(sample_data)
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, node_color=sample_data.y.numpy(), 
            node_size=500, with_labels=True, cmap=plt.cm.Set3)
    plt.title('Graph Structure')
    
    plt.subplot(1, 3, 2)
    sns.heatmap(adj.numpy(), annot=True, cmap='Blues', cbar=True)
    plt.gca().invert_yaxis() 
    plt.title('Adjacency Matrix')
    
    plt.subplot(1, 3, 3)
    plt.imshow(sample_data.x.numpy(), cmap='viridis', aspect='auto')
    plt.colorbar(label='Feature Value')
    plt.title('Node Features')
    plt.xlabel('Feature Dimension')
    plt.ylabel('Node Index')
    
    plt.tight_layout()
    plt.show()
    
    # 2. Train GCN on sample data
    print("\n2. Training GCN on sample data...")
    gcn_model = SimpleGCN(input_dim=3, hidden_dim=8, output_dim=2)
    losses, embeddings_history, _ = train_gnn(gcn_model, sample_data, epochs=100)
    
    plot_training_progress(losses, embeddings_history, sample_data.y, "GCN")
    plt.show()
    
    # 3. Compare architectures on Karate Club
    print("\n3. Comparing architectures on Karate Club dataset...")
    plt_comparison, results = compare_gnn_architectures()
    plt_comparison.show()
    
    # 4. Visualize Karate Club graph
    print("\n4. Visualizing Karate Club graph structure...")
    karate_data = create_karate_club_graph()
    visualize_graph(karate_data, "Karate Club Graph", karate_data.y)
    plt.show()
    
    print("\n=== Tutorial Completed Successfully! ===")
    print("You've learned:")
    print("✓ Graph data structures and adjacency matrices")
    print("✓ Graph Convolutional Networks (GCN) from scratch")
    print("✓ Graph Attention Networks (GAT) basics")
    print("✓ Training GNNs and visualizing results")
    print("✓ Comparing different architectures")
    print("✓ Working with real graph datasets (Karate Club)")

# Additional utility functions
def analyze_graph_properties(data):
    """Analyze basic graph properties"""
    G = graph_to_networkx(data)
    
    print("Graph Properties:")
    print(f"• Nodes: {G.number_of_nodes()}")
    print(f"• Edges: {G.number_of_edges()}")
    print(f"• Average degree: {2 * G.number_of_edges() / G.number_of_nodes():.2f}")
    print(f"• Clustering coefficient: {nx.average_clustering(G):.3f}")
    print(f"• Is connected: {nx.is_connected(G)}")
    
    if nx.is_connected(G):
        print(f"• Average path length: {nx.average_shortest_path_length(G):.2f}")

def create_grid_graph(rows=5, cols=5):
    """Create a grid graph for testing"""
    G = nx.grid_2d_graph(rows, cols)
    
    # Convert node labels to integers
    mapping = {node: i for i, node in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, mapping)
    
    # Convert to our format
    edge_list = list(G.edges())
    edge_index = torch.tensor([[u, v] for u, v in edge_list] + 
                             [[v, u] for u, v in edge_list], dtype=torch.long).t()
    
    num_nodes = G.number_of_nodes()
    x = torch.randn(num_nodes, 4)  # Random features
    y = torch.randint(0, 2, (num_nodes,))  # Binary classification
    
    return GraphData(x, edge_index, y)

# Run the demo
if __name__ == "__main__":
    print("Required packages: torch, numpy, matplotlib, networkx, seaborn, scikit-learn")
    print("Install with: pip install torch numpy matplotlib networkx seaborn scikit-learn")
    print()
    
    main_demo()