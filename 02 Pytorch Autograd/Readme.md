# 02 PyTorch Autograd

Welcome to the **PyTorch Autograd** section! This folder contains a Jupyter notebook focused on understanding automatic differentiation — a key feature in PyTorch for training neural networks.

## 🌟 What is Autograd?
Autograd is PyTorch's automatic differentiation engine. It tracks all operations on tensors that have `requires_grad=True` and automatically computes gradients for backpropagation.

## 🚀 Topics Covered

1. **Introduction to Autograd**
   - The concept of automatic differentiation
   - Why gradients are essential for training deep learning models

2. **Tracking Tensor Operations**
   - Tensors with `requires_grad=True`
   - Building a computation graph

3. **Backward Propagation**
   - Using the `.backward()` function to compute gradients
   - Understanding the gradient of simple scalar tensors

4. **Gradient Calculation**
   - Accessing gradients via `tensor.grad`
   - Zeroing gradients with `optimizer.zero_grad()`

5. **Detaching Tensors**
   - Stopping a tensor from tracking history using `.detach()`
   - Working in `torch.no_grad()` for inference

6. **Computation Graphs**
   - How PyTorch dynamically builds computation graphs
   - Visualization and optimization of these graphs

## 🏎️ Running the Notebook

1. Ensure PyTorch is installed:
   ```bash
   pip install torch
   ```

2. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

3. Open the **Pytorch Autograd** notebook and experiment with automatic differentiation!

## 🔥 Why Learn Autograd?
Understanding autograd is crucial because it:
- Automates backpropagation for deep learning models
- Simplifies gradient computation for complex models
- Helps debug and optimize training pipelines

---

Let’s demystify how PyTorch handles gradients and train smarter models! 💡