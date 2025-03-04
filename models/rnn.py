import numpy as np
from models.base_model import BaseRNN

class VanillaRNN(BaseRNN):
    def __init__(self, input_size, hidden_size):
        super().__init__(input_size, hidden_size)
        # Xavier initialization
        self.Wxh = np.random.randn(hidden_size, input_size) * np.sqrt(2.0 / (input_size + hidden_size))
        self.Whh = np.random.randn(hidden_size, hidden_size) * np.sqrt(2.0 / (hidden_size + hidden_size))
        self.bh = np.zeros((hidden_size, 1))
        self.Why = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / (hidden_size + input_size))  # Output layer
        self.by = np.zeros((input_size, 1))  # Output bias
        self.h_next = None
        self.x = None
        self.h_prev = None

    def forward(self, x, h_prev):
        """
        Forward pass for one timestep.
        x: input at current timestep (shape: input_size x batch_size)
        h_prev: previous hidden state (shape: hidden_size x batch_size)
        """
        self.x = x  # Shape: [input_size, batch_size]
        self.h_prev = h_prev  # Shape: [hidden_size, batch_size]
        self.h_next = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h_prev) + self.bh)
        return self.h_next

    def output(self, h):
        """Map hidden state to output (vocabulary size)"""
        return np.dot(self.Why, h) + self.by]  # Shape: [input_size, batch_size]

    def backward(self, dy):
        """
        Backward pass for one timestep.
        dy: Gradient of the loss w.r.t. the output (shape: input_size x batch_size)
        """
        print(f"dy shape: {dy.shape}, Why.T shape: {self.Why.T.shape}")  # Debug
        # Gradient through the output layer
        if dy.shape[0] != self.Why.shape[0]:
            raise ValueError(
                f"Shape mismatch: dy has shape {dy.shape}, "
                f"but Why.T expects {self.Why.T.shape[1]} features."
            )

        # Gradient through the output layer
        dh_next = np.dot(self.Why.T, dy)  # Now shape: [hidden_size, batch_size]
        
        # Gradient through tanh
        dh_raw = (1 - self.h_next**2) * dh_next  # Shape: [hidden_size, batch_size]
        
        # Gradients for weights and biases
        dWxh = np.dot(dh_raw, self.x.T)  # Shape: [hidden_size, input_size]
        dWhh = np.dot(dh_raw, self.h_prev.T)  # Shape: [hidden_size, hidden_size]
        dbh = np.sum(dh_raw, axis=1, keepdims=True)  # Shape: [hidden_size, 1]
        
        # Gradients for output layer
        dWhy = np.dot(dy, self.h_next.T)  # Shape: [input_size, hidden_size]
        dby = np.sum(dy, axis=1, keepdims=True)  # Shape: [input_size, 1]
        
        # Gradient for previous hidden state
        dh_prev = np.dot(self.Whh.T, dh_raw)  # Shape: [hidden_size, batch_size]
        
        # Clipping gradients to prevent explosions
        dWxh = np.clip(dWxh, -1, 1)
        dWhh = np.clip(dWhh, -1, 1)
        dbh = np.clip(dbh, -1, 1)
        dWhy = np.clip(dWhy, -1, 1)
        dby = np.clip(dby, -1, 1)
        
        return dWxh, dWhh, dbh, dWhy, dby, dh_prev

    def zero_grad(self):
        """Reset gradients."""
        self.dWxh = np.zeros_like(self.Wxh)
        self.dWhh = np.zeros_like(self.Whh)
        self.dbh = np.zeros_like(self.bh)
        self.dWhy = np.zeros_like(self.Why)
        self.dby = np.zeros_like(self.by)

    def init_hidden(self, batch_size):
        """Initialize hidden state."""
        return np.zeros((self.hidden_size, batch_size))