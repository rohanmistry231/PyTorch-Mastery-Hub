# 06 Building ANN using PyTorch

Welcome to the **Building ANN using PyTorch** section! This folder contains a Jupyter notebook that guides you through constructing an Artificial Neural Network (ANN) from scratch using PyTorch, with the Fashion MNIST dataset.

## 🧠 What is an Artificial Neural Network (ANN)?
ANNs are computational models inspired by the human brain, composed of layers of interconnected nodes (neurons). They are essential for solving complex problems like image classification, natural language processing, and more.

## 🚀 Topics Covered

1. **Introduction to ANNs**
   - Understanding neural networks and their components
   - The role of input, hidden, and output layers

2. **Loading the Fashion MNIST Dataset**
   - Using `torchvision.datasets` to load Fashion MNIST
   - Preprocessing data with transforms

3. **Building the ANN Model**
   - Defining the model architecture using `torch.nn.Module`
   - Adding layers: `nn.Linear`, `nn.ReLU`, `nn.Softmax`
   - Implementing the `forward()` method

4. **Loss Function and Optimizer**
   - Using `CrossEntropyLoss` for multi-class classification
   - Initializing optimizers like `torch.optim.SGD` and `Adam`

5. **Training the ANN**
   - Forward pass: computing predictions
   - Calculating loss
   - Backpropagation using `loss.backward()`
   - Optimizing weights with `optimizer.step()`
   - Zeroing gradients with `optimizer.zero_grad()`

6. **Evaluating Model Performance**
   - Switching to evaluation mode with `model.eval()`
   - Calculating accuracy

7. **Visualizing Training Progress**
   - Plotting loss and accuracy using Matplotlib

## 🏎️ Running the Notebook

1. Ensure PyTorch and torchvision are installed:
   ```bash
   pip install torch torchvision
   ```

2. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

3. Open the **ann_fashion_mnist_pytorch.ipynb** notebook and start building your ANN!

## 🔥 Why Learn ANN in PyTorch?
Building ANNs in PyTorch helps you:
- Master the foundations of deep learning
- Create and train neural networks from scratch
- Prepare for advanced concepts like CNNs and RNNs

---

Let’s build intelligent models step by step! 🌟