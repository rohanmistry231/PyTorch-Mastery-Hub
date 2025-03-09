# 10 Building a CNN using PyTorch

Welcome to the **Building a CNN using PyTorch** section! This folder contains a Jupyter notebook guiding you through creating a Convolutional Neural Network (CNN) for the Fashion MNIST dataset using PyTorch.

## 📸 What is a Convolutional Neural Network (CNN)?
CNNs are a type of neural network designed for processing structured grid data like images. They use convolutional layers to automatically and adaptively learn spatial hierarchies of features from input images.

## 🚀 Topics Covered

1. **Introduction to CNNs**
   - Understanding the convolution operation
   - Importance of feature maps and kernels
   - Pooling layers for dimensionality reduction

2. **Fashion MNIST Dataset**
   - Loading Fashion MNIST using `torchvision.datasets`
   - Preprocessing with data transforms

3. **Building a CNN Model**
   - Using `torch.nn.Module` to define CNN layers:
     - `nn.Conv2d` (convolutional layer)
     - `nn.ReLU` (activation function)
     - `nn.MaxPool2d` (pooling layer)
     - `nn.Linear` (fully connected layer)
   - Implementing the `forward()` method

4. **Loss Function and Optimizer**
   - Configuring `CrossEntropyLoss` for multi-class classification
   - Using optimizers like `torch.optim.Adam`

5. **Training the CNN**
   - Forward and backward passes
   - Computing loss and updating weights
   - Using GPU acceleration with `.to(device)`

6. **Model Evaluation**
   - Switching to evaluation mode using `model.eval()`
   - Calculating accuracy and visualizing predictions

7. **Visualizing Filters and Feature Maps**
   - Inspecting learned filters (kernels)
   - Visualizing intermediate feature maps

## 🏎️ Running the Notebook

1. Ensure PyTorch and torchvision are installed:
   ```bash
   pip install torch torchvision
   ```

2. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

3. Open **cnn_fashion_mnist_pytorch_gpu.ipynb** and start building your CNN!

## ✅ Why Learn CNNs in PyTorch?
Mastering CNNs is crucial for:
- Developing computer vision models for image classification, object detection, and segmentation
- Understanding feature extraction and hierarchical learning
- Building state-of-the-art vision applications

---

Let’s unlock the power of convolutional networks! 🚀