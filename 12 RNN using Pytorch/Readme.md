# 12 RNN using PyTorch

Welcome to the **RNN using PyTorch** section! This folder contains a Jupyter notebook focused on building a **Recurrent Neural Network (RNN)** for a Question-Answering (QA) system using PyTorch.

## 🔄 What is an RNN?
Recurrent Neural Networks (RNNs) are a class of neural networks designed to process sequential data by maintaining a hidden state that captures information from previous time steps. They are widely used for tasks like language modeling, text generation, and time series prediction.

## 🚀 Topics Covered

1. **Introduction to RNNs**
   - Understanding the need for RNNs in sequential data processing
   - Difference between feedforward networks and RNNs

2. **QA Dataset Preparation**
   - Using the **100 Unique QA Dataset**
   - Preprocessing text data (tokenization, padding, etc.)

3. **Building the RNN Model**
   - Using `torch.nn.RNN` to create a simple RNN
   - Defining layers like:
     - `nn.Embedding` (word embeddings)
     - `nn.RNN` (recurrent layer)
     - `nn.Linear` (output layer)
   - Implementing the `forward()` method

4. **Loss Function and Optimizer**
   - Configuring `CrossEntropyLoss` for text classification
   - Using `torch.optim.Adam` for optimization

5. **Training the RNN**
   - Forward and backward passes
   - Calculating loss and updating weights
   - Using GPU acceleration with `.to(device)`

6. **Model Evaluation**
   - Evaluating model performance with accuracy metrics
   - Visualizing loss and accuracy plots

7. **Making Predictions**
   - Feeding new questions into the trained model
   - Interpreting RNN outputs

## 🏎️ Running the Notebook

1. Ensure PyTorch is installed:
   ```bash
   pip install torch
   ```

2. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

3. Open the **pytorch_rnn_based_qa_system.ipynb** notebook to train and test the RNN model!

## ✅ Why Learn RNNs in PyTorch?
Mastering RNNs allows you to:
- Build models for sequence prediction tasks
- Process natural language data efficiently
- Lay the foundation for more advanced architectures like LSTMs and GRUs

---

Let’s build dynamic models for sequential data with PyTorch RNNs! 🔥