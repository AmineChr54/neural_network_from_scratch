# Neural Network from Scratch

A Python implementation of a neural network built from scratch using NumPy for handwritten digit recognition on the MNIST dataset.

## 📋 Overview

This project implements a two-layer neural network entirely from scratch without using deep learning frameworks like TensorFlow or PyTorch. The network is designed to classify handwritten digits (0-9) from the MNIST dataset, demonstrating fundamental concepts of neural networks including forward propagation, backpropagation, and gradient descent.

## ✨ Features

- **Custom Neural Network Architecture**: Two-layer neural network with configurable parameters
- **ReLU Activation**: Uses ReLU (Rectified Linear Unit) activation for the hidden layer
- **Softmax Output**: Softmax activation for multi-class classification
- **Backpropagation**: Custom implementation of the backpropagation algorithm
- **Training Visualization**: Tools for visualizing predictions and training results
- **MNIST Dataset Support**: Designed to work with the MNIST handwritten digits dataset

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/AmineChr54/neural_network_from_scratch.git
cd neural_network_from_scratch
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## 📦 Requirements

- numpy - For numerical computations and matrix operations
- pandas - For data loading and manipulation
- matplotlib - For visualization and plotting
- ipykernel - For Jupyter notebook support

## 📊 Dataset Setup

This project uses the MNIST dataset. You need to:

1. Download the MNIST dataset in CSV format
2. Place the training data in `./data/MNIST/mnist_train.csv`
3. The dataset should have labels in the first column and pixel values (0-255) in subsequent columns

You can download the MNIST dataset from sources like:
- [Kaggle MNIST Dataset](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv)
- [MNIST Original](http://yann.lecun.com/exdb/mnist/)

## 🚀 Usage

### Training the Neural Network

Open and run the `main.ipynb` notebook:

```python
# The main workflow includes:
# 1. Load and preprocess MNIST data
# 2. Split into training and test sets
# 3. Initialize network parameters
# 4. Train the model
# 5. Evaluate accuracy
```

### Visualizing Results

The project includes utilities for plotting:

```python
from plot_utils import plot_number

# Visualize a prediction
plot_number(image_index, guessed_number)
```

Alternatively, use the `plotting.ipynb` notebook for more visualization options.

## 🏗️ Project Structure

```
neural_network_from_scratch/
├── main.ipynb           # Main training notebook
├── plotting.ipynb       # Visualization notebook
├── plot_utils.py        # Utility functions for plotting
├── requirements.txt     # Project dependencies
├── readme.md           # This file
└── .gitignore          # Git ignore rules
```

## 🧮 Network Architecture

- **Input Layer**: 784 neurons (28×28 pixel images flattened)
- **Hidden Layer**: 10 neurons with ReLU activation
- **Output Layer**: 10 neurons with Softmax activation (one for each digit 0-9)

### Key Functions

- `init_variables()`: Initialize weights and biases
- `forward_prop()`: Forward propagation through the network
- `backward_prop()`: Backpropagation to compute gradients
- `update_variables()`: Update weights and biases using gradient descent
- `train_model()`: Main training loop

## 📈 Model Training

The training process includes:
1. Random shuffling of data
2. Splitting data into training and test sets
3. Forward propagation to make predictions
4. Backpropagation to compute gradients
5. Parameter updates using gradient descent
6. Accuracy evaluation at regular intervals

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Submit pull requests

## 📝 License

This project is open source and available for educational purposes.

## 🎓 Learning Objectives

This project demonstrates:
- Implementation of neural network fundamentals from scratch
- Matrix operations for efficient computation
- Gradient descent optimization
- Activation functions (ReLU, Softmax)
- One-hot encoding for multi-class classification
- Training and evaluation workflows

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Note**: This is an educational project designed to understand the inner workings of neural networks. For production use, consider using established frameworks like TensorFlow or PyTorch.
