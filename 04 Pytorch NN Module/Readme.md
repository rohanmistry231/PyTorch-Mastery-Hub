# 04 PyTorch NN Module

Welcome to the **PyTorch NN Module** section! This folder contains a Jupyter notebook focused on using the `torch.nn` module to build and train neural networks efficiently.

## 🌟 What is the NN Module?
The `torch.nn` module is a core library in PyTorch that provides pre-built classes for creating neural networks, including layers, activation functions, loss functions, and more. It abstracts away the complexity of manually defining network components, allowing for faster experimentation and model development.

## 🚀 Topics Covered

1. **Introduction to the NN Module**
   - What is `torch.nn` and why use it?
   - Key benefits of using pre-built modules

2. **Building Neural Networks**
   - Creating models by subclassing `nn.Module`
   - Defining layers (e.g., `nn.Linear`, `nn.Conv2d`, `nn.LSTM`)
   - Implementing the `forward()` method

3. **Activation Functions**
   - Using built-in activation functions: `nn.ReLU`, `nn.Sigmoid`, `nn.Tanh`
   - Adding non-linearity to neural networks

4. **Loss Functions**
   - Common loss functions: `nn.CrossEntropyLoss`, `nn.MSELoss`, `nn.NLLLoss`
   - Calculating loss for optimization

5. **Optimizers**
   - Introduction to `torch.optim`
   - Optimizers like SGD, Adam, and RMSprop
   - Linking optimizers to model parameters using `model.parameters()`

6. **Sequential Containers**
   - Stacking layers using `nn.Sequential`
   - Simplifying model definitions

7. **Regularization and Dropout**
   - Using `nn.Dropout` and `nn.BatchNorm2d`
   - Preventing overfitting and improving generalization

## 🏎️ Running the Notebook

1. Ensure PyTorch is installed:
   ```bash
   pip install torch
   ```

2. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

3. Open the **PyTorch NN Module** notebook and start building neural networks!

## 🔥 Why Learn the NN Module?
Mastering the `torch.nn` module is essential because it:
- Simplifies neural network creation
- Integrates seamlessly with PyTorch’s autograd system
- Provides flexibility for both simple and complex models

---

Let’s build powerful neural networks with PyTorch! 🧠