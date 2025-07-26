#  Graph Neural Networks (GNN) Educational Framework

A comprehensive educational codebase designed to teach Graph Neural Networks through hands-on implementation and interactive visualization. This tutorial builds GNNs from scratch using pure PyTorch, making complex concepts accessible to students and researchers.

##  Educational Objectives

This framework was created to help students understand:
- **Graph Theory Fundamentals**: Adjacency matrices, graph properties, and normalization
- **GNN Architecture**: How convolution works on non-Euclidean graph data
- **Attention Mechanisms**: Graph attention networks and message passing
- **Training Dynamics**: How GNNs learn node representations
- **Practical Implementation**: Building GNNs without external graph libraries

##  Quick Start

### Prerequisites
```bash
pip install torch numpy matplotlib networkx seaborn scikit-learn
```

### Running the Tutorial
```bash
python main.py
```

That's it! The tutorial will automatically run through all demonstrations with interactive visualizations.

##  What You'll Learn

### 1. **Graph Data Structures**
- Custom graph data representation
- Edge index to adjacency matrix conversion
- Graph normalization techniques
- NetworkX integration for visualization

### 2. **GNN Architectures (From Scratch)**
- **Graph Convolutional Networks (GCN)**
  - Manual implementation of graph convolution
  - Adjacency matrix normalization
  - Layer-wise propagation
  
- **Graph Attention Networks (GAT)**
  - Attention mechanism computation
  - Multi-head attention (simplified)
  - Learnable attention weights

### 3. **Interactive Visualizations**
-  Graph structure plotting
-  Adjacency matrix heatmaps
-  Node feature visualization
-  t-SNE embedding evolution
-  Training progress tracking
-  Architecture comparison

### 4. **Real-World Datasets**
- **Karate Club Graph**: Classic benchmark dataset
- **Synthetic Communities**: Generated graph structures
- **Custom Examples**: Educational demonstrations

## Tutorial Flow

When you run `python main.py`, you'll experience:

1. **Graph Basics** 
   - Understanding graph data structures
   - Visualizing adjacency matrices
   - Node features representation

2. **GCN Implementation**
   - Building graph convolution from scratch
   - Training on sample data
   - Embedding evolution visualization

3. **Architecture Comparison** 
   - GCN vs GAT performance
   - Side-by-side visualizations
   - Accuracy comparisons

4. **Real Dataset Analysis** 
   - Karate Club graph exploration
   - Community detection visualization
   - Final results interpretation

## Sample Outputs

The tutorial generates several types of visualizations:

- **Graph Structure**: Node-link diagrams with community colors
- **Adjacency Matrices**: Heatmaps showing connection patterns  
- **Embedding Evolution**: How node representations change during training
- **Training Curves**: Loss progression and accuracy metrics
- **t-SNE Plots**: 2D projections of high-dimensional embeddings

## Key Features

### **No Complex Dependencies**
- Pure PyTorch implementation
- No torch_geometric required
- Standard scientific Python stack only

### **Educational Focus**
- Step-by-step explanations
- Mathematical foundations exposed
- Visual learning approach
- Modular, readable code

### **Complete Framework**
- Data loading and preprocessing
- Model implementation from scratch
- Training and evaluation pipelines
- Comprehensive visualizations

### **Hands-On Learning**
- Interactive plots and figures
- Real-time training visualization
- Comparative analysis tools
- Extensible codebase

## Learning Outcomes

After completing this tutorial, students will be able to:

-  Understand the mathematical foundations of graph neural networks
-  Implement GCN and GAT layers from scratch
-  Visualize and interpret graph data and embeddings
-  Train GNN models on real datasets
-  Compare different GNN architectures
-  Extend the framework for custom applications

## Customization

The codebase is designed for easy extension:

```python
# Add your own GNN layer
class CustomGNNLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # Your implementation here
        
# Create custom datasets
def create_my_graph():
    # Your graph creation logic
    return GraphData(x, edge_index, y)

# Add to comparison
models['MyGNN'] = CustomGNNLayer(...)
```

## Educational Use Cases

Perfect for:
- **University Courses**: Machine Learning, Deep Learning, Graph Theory
- **Research Groups**: Understanding GNN fundamentals
- **Self-Study**: Hands-on GNN learning
- **Workshops**: Interactive GNN demonstrations
- **Tutorials**: Step-by-step GNN education

## Contributing

This is an educational resource! Contributions welcome:
- Additional GNN architectures
- More visualization types
- New datasets
- Educational improvements
- Bug fixes and optimizations

## References

This tutorial implements concepts from:
- Kipf & Welling (2017): "Semi-Supervised Classification with Graph Convolutional Networks"
- Veličković et al. (2018): "Graph Attention Networks"

## License

MIT License - Feel free to use for educational purposes!

## Ready to Start?

```bash
python main.py
```
