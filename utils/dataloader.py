import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from collections import Counter

class DataLoader:
    def __init__(self, seq_length=50):
        self.seq_length = seq_length
        self.char_to_idx = {}
        self.idx_to_char = {}
        
    def load_text_data(self, file_path):
        with open(file_path, 'r') as f:
            text = f.read()
        chars = sorted(list(set(text)))
        self.char_to_idx = {ch:i for i,ch in enumerate(chars)}
        self.idx_to_char = {i:ch for i,ch in enumerate(chars)}
        return self.text_to_seq(text)
    
    def text_to_seq(self, text):
        return np.array([self.char_to_idx[ch] for ch in text])
    
    def create_batches(self, data, batch_size):
        num_batches = len(data) // (batch_size * self.seq_length)
        data = data[:num_batches * batch_size * self.seq_length]
        x = data.reshape(batch_size, -1, self.seq_length)
        y = np.zeros_like(x)
        y[:, :-1] = x[:, 1:]
        return x, y

class ClassificationLoader:
    def __init__(self, max_words=10000, max_len=500):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = imdb.load_data(num_words=max_words)
        self.max_words = max_words
        self.max_len = max_len
        self.x_train = pad_sequences(self.x_train, maxlen=max_len)
        self.x_test = pad_sequences(self.x_test, maxlen=max_len)

    def get_batches(self, batch_size):
        num_batches = len(self.x_train) // batch_size
        for i in range(num_batches):
            x_batch = self.x_train[i*batch_size:(i+1)*batch_size]
            y_batch = self.y_train[i*batch_size:(i+1)*batch_size]
            yield x_batch, y_batch