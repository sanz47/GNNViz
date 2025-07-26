# 📊 Graph Neural Networks (GNN) Educational Framework

A comprehensive educational codebase designed to teach Graph Neural Networks through hands-on implementation and interactive visualization. This tutorial builds GNNs from scratch using pure PyTorch, making complex concepts accessible to students and researchers.

## 🎯 Educational Objectives

This framework was created to help students understand:
- **Graph Theory Fundamentals**: Adjacency matrices, graph properties, and normalization
- **GNN Architecture**: How convolution works on non-Euclidean graph data
- **Attention Mechanisms**: Graph attention networks and message passing
- **Training Dynamics**: How GNNs learn node representations
- **Practical Implementation**: Building GNNs without external graph libraries

## 🚀 Quick Start

### Prerequisites
```bash
pip install torch numpy matplotlib networkx seaborn scikit-learn
```

### Running the Tutorial
```bash
python main.py
```

That's it! The tutorial will automatically run through all demonstrations with interactive visualizations.

## 📁 Project Structure

```
📦 GNN-Educational-Framework
├── 📄 main.py              # Main tutorial script - RUN THIS FILE
├── 📄 README.md            # This file
└── 📊 outputs/             # Generated plots and visualizations
```

## 🧠 What You'll Learn

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
- 📈 Graph structure plotting
- 🔥 Adjacency matrix heatmaps
- 🎨 Node feature visualization
- 📍 t-SNE embedding evolution
- 📊 Training progress tracking
- ⚖️ Architecture comparison

### 4. **Real-World Datasets**
- **Karate Club Graph**: Classic benchmark dataset
- **Syntheti
