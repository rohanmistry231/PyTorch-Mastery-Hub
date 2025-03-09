# 07 Neural Network Training on GPU

Welcome to the **Neural Network Training on GPU** section! This folder contains a Jupyter notebook demonstrating how to accelerate neural network training using GPU with PyTorch.

## ⚡ Why Train on GPU?
Training neural networks on a GPU significantly boosts performance, especially for large datasets and complex models. PyTorch makes it simple to transfer data and models to the GPU for efficient computations.

## 🚀 Topics Covered

1. **Introduction to GPU Acceleration**
   - Why GPUs are faster for deep learning
   - Key differences between CPU and GPU processing

2. **Checking GPU Availability**
   - Using `torch.cuda.is_available()` to check GPU support
   - Listing available GPUs with `torch.cuda.device_count()`

3. **Moving Tensors to GPU**
   - Transferring tensors to GPU with `.to(device)`
   - Ensuring models and data reside on the same device

4. **Building an ANN for Fashion MNIST**
   - Using `torch.nn.Module` to define the architecture
   - Implementing layers like `nn.Linear`, `nn.ReLU`, and `nn.Softmax`

5. **Loss Function and Optimizer**
   - Configuring `CrossEntropyLoss` for classification tasks
   - Using `torch.optim.Adam` for optimization

6. **Training the ANN on GPU**
   - Forward pass and backward pass on GPU
   - Computing gradients and updating weights with GPU acceleration

7. **Model Evaluation**
   - Switching to evaluation mode with `model.eval()`
   - Running inference on GPU using `torch.no_grad()`

8. **Handling Multiple GPUs**
   - Using `DataParallel` for multi-GPU training

## 🏎️ Running the Notebook

1. Ensure PyTorch is installed with CUDA support:
   ```bash
   pip install torch torchvision
   ```

2. Verify GPU access:
   ```python
   import torch
   print("CUDA available:", torch.cuda.is_available())
   print("Number of GPUs:", torch.cuda.device_count())
   ```

3. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

4. Open **ann_fashion_mnist_pytorch_gpu.ipynb** and start training models with GPU power!

## 🔥 Why Learn GPU Training in PyTorch?
Mastering GPU training helps you:
- Drastically reduce model training time
- Handle larger datasets and complex architectures
- Prepare for distributed training with multiple GPUs or TPUs

---

Let’s harness the power of GPUs to train smarter and faster! 🚀