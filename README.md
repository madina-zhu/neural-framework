# Neural Framework - Custom Deep Learning Framework

A lightweight neural network framework built from scratch with NumPy, featuring automatic differentiation, modular architecture, and a no-code web interface.

## 👥 Authors
- **Катя** - Core framework (autograd, layers, optimizers, losses)
- **Мадина** - Data module, examples, documentation, no-code interface

## ✨ Features

### Core Framework (Катя)
- ✅ Custom autograd engine with computation graph
- ✅ Modular layers (Linear, Sequential)
- ✅ Activation functions (ReLU, Sigmoid, Softmax, Tanh)
- ✅ Loss functions (MSE, CrossEntropy)
- ✅ Optimizers (SGD, Momentum, Adam)
- ✅ Gradient clipping
- ✅ Flexible training loop with callbacks

### Data Module (Мадина)
- ✅ Dataset abstraction with easy loading
- ✅ DataLoader with batching, shuffling, mapping
- ✅ Built-in datasets (MNIST, Iris, California Housing)
- ✅ Data transformations (normalization, one-hot, flatten)
- ✅ Train/validation split utilities

### Examples & Demos
- ✅ MNIST digit classification (92%+ accuracy)
- ✅ Iris classification and regression
- ✅ California Housing regression

### No-Code Web Interface
- ✅ Streamlit-based interactive playground
- ✅ Visual model configuration
- ✅ Real-time training visualization
- ✅ Export trained models

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/yourusername/neural-framework.git
cd neural-framework
pip install -r requirements.txt