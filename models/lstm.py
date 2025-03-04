import numpy as np
from models.base_model import BaseRNN

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class LSTM(BaseRNN):
    def __init__(self, input_size, hidden_size):
        super().__init__(input_size, hidden_size)
        # Gate weights (input, forget, output, cell)
        self.Wx = np.random.randn(4*hidden_size, input_size) * np.sqrt(2.0/(input_size + hidden_size))
        self.Wh = np.random.randn(4*hidden_size, hidden_size) * np.sqrt(2.0/(hidden_size + hidden_size))
        self.b = np.zeros((4*hidden_size, 1))
        
        # Output layer
        self.Why = np.random.randn(input_size, hidden_size) * np.sqrt(2.0/(hidden_size + input_size))
        self.by = np.zeros((input_size, 1))
        
        # State tracking
        self.h_next = None
        self.c_next = None
        self.x = None
        self.h_prev = None
        self.c_prev = None
        self.gates = None

    def forward(self, x, h_prev, c_prev):
        """
        Forward pass for one timestep
        x: input (input_size, batch_size)
        h_prev: previous hidden state (hidden_size, batch_size)
        c_prev: previous cell state (hidden_size, batch_size)
        """
        # Store for backward pass
        self.x = x
        self.h_prev = h_prev
        self.c_prev = c_prev
        
        # Gates computation
        gates = np.dot(self.Wx, x) + np.dot(self.Wh, h_prev) + self.b
        i, f, o, g = np.split(gates, 4, axis=0)
        
        # Gate activations
        self.i = sigmoid(i)
        self.f = sigmoid(f)
        self.o = sigmoid(o)
        self.g = np.tanh(g)
        
        # Cell state update
        self.c_next = self.f * c_prev + self.i * self.g
        self.h_next = self.o * np.tanh(self.c_next)
        
        return self.h_next, self.c_next

    def output(self, h):
        """Map hidden state to output (vocabulary size)"""
        return np.dot(self.Why, h) + self.by  # Shape: [input_size, batch_size]

    def backward(self, dy):
        """
        Backward pass through LSTM cell
        dy: gradient from output layer (input_size, batch_size)
        Returns gradients for all parameters and previous states
        """
        # Gradient through output layer
        dh_next = np.dot(self.Why.T, dy)  # Shape: [hidden_size, batch_size]
        
        # Gradients through LSTM gates
        do = dh_next * np.tanh(self.c_next) * self.o * (1 - self.o)
        dc = dh_next * self.o * (1 - np.tanh(self.c_next)**2)
        dc += dc  # Add gradient from cell state
        
        df = dc * self.c_prev * self.f * (1 - self.f)
        di = dc * self.g * self.i * (1 - self.i)
        dg = dc * self.i * (1 - self.g**2)
        
        # Combine gate gradients
        dgates = np.concatenate((di, df, do, dg))  # [4*hidden, batch]
        
        # Parameter gradients
        dWx = np.dot(dgates, self.x.T)  # [4*hidden, input]
        dWh = np.dot(dgates, self.h_prev.T)  # [4*hidden, hidden]
        db = np.sum(dgates, axis=1, keepdims=True)  # [4*hidden, 1]
        
        # Output layer gradients
        dWhy = np.dot(dy, self.h_next.T)  # [input, hidden]
        dby = np.sum(dy, axis=1, keepdims=True)  # [input, 1]
        
        # Previous state gradients
        dx = np.dot(self.Wx.T, dgates)  # [input, batch]
        dh_prev = np.dot(self.Wh.T, dgates)  # [hidden, batch]
        dc_prev = self.f * dc  # [hidden, batch]

        # Gradient clipping
        dWx = np.clip(dWx, -1, 1)
        dWh = np.clip(dWh, -1, 1)
        db = np.clip(db, -1, 1)
        dWhy = np.clip(dWhy, -1, 1)
        dby = np.clip(dby, -1, 1)

        return dWx, dWh, db, dWhy, dby, dh_prev, dc_prev

    def zero_grad(self):
        """Reset gradients"""
        self.dWx = np.zeros_like(self.Wx)
        self.dWh = np.zeros_like(self.Wh)
        self.db = np.zeros_like(self.b)
        self.dWhy = np.zeros_like(self.Why)
        self.dby = np.zeros_like(self.by)

    def init_hidden(self, batch_size):
        """Initialize hidden and cell states"""
        return (
            np.zeros((self.hidden_size, batch_size)),  # h0
            np.zeros((self.hidden_size, batch_size))   # c0
        )