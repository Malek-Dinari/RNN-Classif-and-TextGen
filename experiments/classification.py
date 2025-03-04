# experiments/classification.py
import os
import sys
import numpy as np
from ..models.lstm import LSTM
from ..models.rnn import VanillaRNN
from utils.metrics import accuracy
from utils.visualize import TensorBoardLogger
from utils.dataloader import ClassificationLoader




# Adding the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def cross_entropy_loss(y_pred, y_true):
    m = y_true.shape[0]
    log_likelihood = -np.log(y_pred[range(m), y_true])
    return np.sum(log_likelihood) / m

def train_classification(model, data, epochs, batch_size, lr=0.001, seq_length=500):
    logger = TensorBoardLogger('logs/classification')
    for epoch in range(epochs):
        epoch_loss = 0
        for x_batch, y_batch in data.get_batches(batch_size):
            
            # Initialize hidden and cell states
            if isinstance(model, LSTM):
                h, c = model.init_hidden(batch_size)  # LSTM requires both h and c
            else:
                h = model.init_hidden(batch_size)  # RNN only requires h
                c = None  # No cell state for RNN

            model.zero_grad()  # Reset gradients
            
            # Convert to one-hot (shape: [seq_length, batch_size, input_size])
            x_one_hot = np.zeros((seq_length, batch_size, model.input_size))
            for b in range(batch_size):
                x_one_hot[np.arange(seq_length), b, x_batch[b]] = 1
            
            # Forward pass (unroll the RNN over sequence steps)
            outputs = []
            for t in range(seq_length):
                # Reshape x_one_hot[t] to (input_size, batch_size)
                x_t = x_one_hot[t].T  # Transpose to match RNN input shape
                h = model.forward(x_t, h)
                outputs.append(model.output(h))  # Use output layer
            
            # Final output (last timestep)
            y_pred = softmax(outputs[-1].T)  # Shape: [batch_size, input_size]
            loss = cross_entropy_loss(y_pred, y_batch)
            
            # Backward pass (BPTT)
            dy = (y_pred - np.eye(model.input_size)[y_batch]).T # Gradient of cross-entropy (shape: [input_size, batch_size])
            for t in reversed(range(seq_length)):
                if isinstance(model, LSTM):
                    # LSTM backward pass
                    dWx, dWh, db, dWhy, dby, dh_prev, dc_prev = model.backward(dy)
                    dy = dh_prev  # Propagate gradient to previous hidden state
                    c = dc_prev  # Propagate gradient to previous cell state
                else:
                    # RNN backward pass
                    dWxh, dWhh, dbh, dWhy, dby, dh_prev = model.backward(dy)
                    dy = dh_prev  # Propagate gradient to previous hidden state
                
                # Update weights
                if isinstance(model, LSTM):
                    model.Wx -= lr * dWx
                    model.Wh -= lr * dWh
                    model.b -= lr * db
                else:
                    model.Wxh -= lr * dWxh
                    model.Whh -= lr * dWhh
                    model.bh -= lr * dbh
                
                # Update output layer weights
                model.Why -= lr * dWhy
                model.by -= lr * dby
            
            epoch_loss += loss
        
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")

if __name__ == "__main__":
    # Load data
    loader = ClassificationLoader()
    batch_size = 32

    # Initialize models
    rnn = VanillaRNN(input_size=10000, hidden_size=128)
    lstm = LSTM(input_size=10000, hidden_size=128)

    # Train models
    train_classification(rnn, loader, epochs=40, batch_size=batch_size)
    train_classification(lstm, loader, epochs=40, batch_size=batch_size)