# RNN-Classif-and-TextGen

## Overview
This repository is dedicated to exploring and contrasting the capabilities of Recurrent Neural Networks (RNNs) and Long Short-Term Memory networks (LSTMs) for classification and text generation tasks. The project includes:

1. A **vanilla RNN** implementation to highlight the core functionality and limitations of basic recurrent networks.
2. An **LSTM implementation from scratch**, showcasing the added benefit of long short-term memory in handling long-range dependencies and mitigating the vanishing gradient problem.
3. A **recap of gradient descent** algorithms with a focus on:
   - The issues of exploding and vanishing gradients.
   - Techniques like gradient clipping and proper weight initialization to address these problems.

## Objectives
1. Understand and implement the fundamental workings of RNNs and LSTMs.
2. Provide insights into the challenges of training RNN-based models.
3. Demonstrate the advantages of LSTMs over vanilla RNNs.
4. Implement gradient descent variations and discuss practical solutions to gradient-related issues.

## Directory Structure
```plaintext
RNN-Classif-and-TextGen/
├── rnn_from_scratch.py       # Implementation of a vanilla RNN.
├── lstm_from_scratch.py      # Implementation of an LSTM from scratch.
├── gradient_descent.py       # Recap of gradient descent and gradient-related problems.
├── data/                     # Example datasets for classification and text generation.
├── models/                   # Saved model weights.
└── README.md                 # Project overview and details.
```
