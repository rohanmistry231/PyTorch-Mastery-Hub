# 01 Tensors in PyTorch

Welcome to the **Tensors in PyTorch** section! This folder contains a Jupyter notebook exploring the fundamentals of tensors — the core data structure in PyTorch.

## 📚 What are Tensors?
Tensors are multi-dimensional arrays, similar to NumPy arrays, but with the added benefit of GPU acceleration. They are the foundation for all computations in PyTorch.

## 🚀 Topics Covered

1. **Introduction to PyTorch**
   - Overview of PyTorch: Open-source deep learning library by Meta AI.
   - Combines Python’s simplicity with Torch’s high-performance tensor operations.

2. **PyTorch Release Timeline**
   - PyTorch 0.1 (2017): Dynamic computation graph, seamless Python library integration.
   - PyTorch 1.0 (2018): TorchScript, Caffe2 integration for research-to-production transitions.
   - PyTorch 1.x Series: Distributed training, ONNX, quantization, expanded ecosystem (torchvision, torchtext, torchaudio).
   - PyTorch 2.0: Performance boosts, modern hardware optimization (TPUs, AI chips).

3. **Core Features of PyTorch**
   - Tensor computations
   - GPU acceleration
   - Dynamic computation graph
   - Automatic differentiation
   - Distributed training
   - Interoperability with other libraries

4. **Tensors in Action**
   - Creating tensors: `torch.tensor()`
   - Basic tensor operations (addition, subtraction, multiplication)
   - Reshaping tensors
   - Moving tensors to GPU (`.to(device)`)

5. **PyTorch vs TensorFlow**
   - A quick comparison of their computational graphs, flexibility, and community support.

6. **PyTorch Core Modules**
   - `torch`: Core tensor library
   - `torch.nn`: Neural network module
   - `torch.autograd`: Automatic differentiation
   - `torch.optim`: Optimization algorithms
   - `torch.utils.data`: Data handling (datasets, data loaders)

## 🏎️ Running the Notebook

1. Ensure you have PyTorch installed:
   ```bash
   pip install torch
   ```

2. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

3. Open the **Tensors in PyTorch** notebook and start experimenting!

## 🔥 Why Learn Tensors?
Understanding tensors is crucial for building deep learning models, as they:
- Store input data, weights, and gradients
- Facilitate matrix operations needed for backpropagation
- Enable seamless GPU computations for faster training

---

Happy coding! Let’s master PyTorch one tensor at a time. ✨