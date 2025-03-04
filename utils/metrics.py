# utils/metrics.py
import numpy as np

def accuracy(y_pred, y_true):
    """
    Calculate accuracy for classification tasks.
    y_pred: Predicted labels (batch_size, num_classes)
    y_true: True labels (batch_size,)
    """
    if y_pred.ndim > 1:
        y_pred = np.argmax(y_pred, axis=1)
    return np.mean(y_pred == y_true)

def perplexity(y_pred, y_true):
    """
    Calculate perplexity for text generation tasks.
    y_pred: Predicted probabilities (batch_size, vocab_size)
    y_true: True labels (batch_size,)
    """
    probs = y_pred[np.arange(len(y_true)), y_true]
    return np.exp(-np.mean(np.log(probs + 1e-10)))