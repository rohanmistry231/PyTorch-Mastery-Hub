# 13 Next Word Predictor using PyTorch

Welcome to the **Next Word Predictor using PyTorch** section! This folder contains a Jupyter notebook that builds a **Next Word Prediction** model using Long Short-Term Memory networks (LSTMs) in PyTorch.

## ✨ What is Next Word Prediction?
Next Word Prediction is a language modeling task where the model predicts the most likely next word given a sequence of previous words. It’s a fundamental concept in Natural Language Processing (NLP) and powers applications like autocomplete and text generation.

## 🚀 Topics Covered

1. **Introduction to Language Modeling**
   - The importance of next word prediction in NLP
   - How language models work

2. **Understanding LSTMs**
   - Why use LSTMs instead of vanilla RNNs
   - The gates in LSTM: Input, Forget, and Output gates

3. **Dataset Preparation**
   - Tokenizing text data
   - Creating input-output pairs for training
   - Padding sequences for batch processing

4. **Building the LSTM Model**
   - Using `torch.nn.LSTM` for sequence modeling
   - Defining key layers:
     - `nn.Embedding` (word embeddings)
     - `nn.LSTM` (recurrent layer)
     - `nn.Linear` (output layer)
   - Implementing the `forward()` method

5. **Loss Function and Optimizer**
   - Using `CrossEntropyLoss` for predicting word probabilities
   - Optimizing with `torch.optim.Adam`

6. **Training the Model**
   - Forward pass, loss calculation, and backpropagation
   - Updating weights and handling gradients

7. **Evaluating Model Performance**
   - Measuring perplexity as an evaluation metric
   - Visualizing training loss and accuracy

8. **Making Predictions**
   - Generating text by predicting words iteratively
   - Sampling strategies for diverse outputs (greedy vs. beam search)

## 🏎️ Running the Notebook

1. Ensure PyTorch is installed:
   ```bash
   pip install torch
   ```

2. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

3. Open **pytorch_lstm_next_word_predictor.ipynb** and start building your next word predictor!

## ✅ Why Learn Next Word Prediction in PyTorch?
Mastering next word prediction helps you:
- Build AI-powered text generation systems
- Enhance auto-complete and chatbot functionalities
- Strengthen your understanding of sequence models like RNNs and LSTMs

---

Let’s build intelligent text generators with PyTorch LSTMs! 🧠