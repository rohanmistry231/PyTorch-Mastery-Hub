# 09 Hyperparameter Tuning the ANN using Optuna

Welcome to the **Hyperparameter Tuning the ANN using Optuna** section! This folder contains a Jupyter notebook demonstrating how to optimize neural network hyperparameters using **Optuna** with PyTorch.

## 🎯 What is Hyperparameter Tuning?
Hyperparameters are the configuration settings outside the model that govern training, such as learning rate, batch size, and the number of hidden layers. Fine-tuning these can dramatically improve model performance.

## 🚀 Why Use Optuna?
Optuna is an open-source hyperparameter optimization framework that automates the search process. It allows you to:
- Define hyperparameters dynamically
- Use advanced search algorithms (Tree-structured Parzen Estimator — TPE)
- Prune unpromising trials early to save compute time

## 🔥 Topics Covered

1. **Introduction to Hyperparameter Tuning**
   - The importance of hyperparameters
   - Grid search vs. random search vs. Bayesian optimization

2. **Setting up Optuna**
   - Installing Optuna:
   ```bash
   pip install optuna
   ```
   - Importing and defining a study object

3. **Defining the Objective Function**
   - Creating an objective function for ANN training
   - Specifying hyperparameters like:
     - Learning rate
     - Batch size
     - Number of hidden layers
     - Optimizer choice

4. **Running Hyperparameter Optimization**
   - Executing multiple trials using `study.optimize()`
   - Setting the number of trials and early stopping criteria

5. **Pruning Unpromising Trials**
   - Using `optuna.pruners.MedianPruner`
   - Integrating pruning with PyTorch training loops

6. **Visualizing Optimization Results**
   - Plotting loss curves for each trial
   - Visualizing hyperparameter relationships with Optuna’s plotting functions:
   ```python
   optuna.visualization.plot_optimization_history(study)
   ```

7. **Best Hyperparameters**
   - Extracting the best set of hyperparameters:
   ```python
   print(study.best_params)
   ```

## 🏎️ Running the Notebook

1. Ensure PyTorch, torchvision, and Optuna are installed:
   ```bash
   pip install torch torchvision optuna
   ```

2. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

3. Open **ann_fashion_mnist_pytorch_gpu_optimized_optuna.ipynb** and start tuning your ANN!

## ✅ Why Learn Hyperparameter Tuning with Optuna?
Mastering hyperparameter tuning helps you:
- Improve model accuracy and reduce training time
- Automate and optimize trial-and-error processes
- Build more robust and generalizable models

---

Let’s unlock the full potential of your ANN with Optuna! 🚀