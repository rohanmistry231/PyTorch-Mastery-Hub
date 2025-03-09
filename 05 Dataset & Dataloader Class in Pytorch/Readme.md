# 05 Dataset & Dataloader Class in PyTorch

Welcome to the **Dataset & Dataloader Class in PyTorch** section! This folder contains a Jupyter notebook focused on how to efficiently load and process data using PyTorch's `Dataset` and `DataLoader` classes.

## 📦 Why Use Dataset and Dataloader?
In PyTorch, datasets and data loaders help manage data processing during training. The `Dataset` class defines how data is loaded, and the `DataLoader` wraps a dataset to handle batching, shuffling, and parallel data loading.

## 🚀 Topics Covered

1. **Introduction to Dataset and DataLoader**
   - Problems with manual data loading (memory inefficiency, slow convergence)
   - Importance of batching and parallelization

2. **Dataset Class**
   - Creating custom datasets by subclassing `torch.utils.data.Dataset`
   - Implementing:
     - `__init__()`: Load data
     - `__len__()`: Return the dataset size
     - `__getitem__(index)`: Fetch a sample (data and label)

3. **DataLoader Class**
   - Wrapping datasets with `DataLoader` for easy batching and shuffling
   - Batch processing, multi-threaded data loading (`num_workers`)
   - Key parameters:
     - `batch_size`
     - `shuffle`
     - `num_workers`
     - `pin_memory`
     - `drop_last`
     - `collate_fn`

4. **Batch and Samplers**
   - Using built-in samplers like:
     - `SequentialSampler`
     - `RandomSampler`
   - Customizing sampling strategies for imbalanced datasets

5. **Parallel Data Loading**
   - Understanding how multiple workers load data in parallel
   - The effect of `num_workers` and batch queueing

6. **Data Transformations**
   - Applying on-the-fly data transformations using `torchvision.transforms`
   - Chaining multiple transforms

7. **Custom Collate Functions**
   - Using `collate_fn` for custom batching logic
   - Handling variable-length sequences and complex data structures

## 🏎️ Running the Notebook

1. Ensure PyTorch is installed:
   ```bash
   pip install torch torchvision
   ```

2. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

3. Open the **Dataset and DataLoader** notebook and start experimenting with efficient data loading!

## 🔥 Why Learn Dataset & Dataloader?
Mastering these concepts allows you to:
- Load and batch large datasets efficiently
- Accelerate model training with parallel data loading
- Handle custom data pipelines and preprocessing

---

Let’s streamline data loading and boost training performance! ⚡