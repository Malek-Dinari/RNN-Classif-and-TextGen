import numpy as np
from models.rnn import VanillaRNN
from models.lstm import LSTM

class TextGenerator:
    def __init__(self, model, idx_to_char):
        self.model = model
        self.idx_to_char = idx_to_char
        
    def generate(self, seed, length=100, temp=0.5):
        h = np.zeros((self.model.hidden_size, 1))
        c = np.zeros((self.model.hidden_size, 1)) if isinstance(self.model, LSTM) else None
        chars = [self.char_to_idx[s] for s in seed]
        
        for _ in range(length):
            x = np.zeros((self.model.input_size, 1))
            x[chars[-1]] = 1
            if isinstance(self.model, LSTM):
                h, c = self.model.forward_step(x, h, c)
            else:
                h = self.model.forward_step(x, h)
            probs = np.exp(h / temp) / np.sum(np.exp(h / temp))
            next_char = np.random.choice(len(self.idx_to_char), p=probs.ravel())
            chars.append(next_char)
        return ''.join([self.idx_to_char[c] for c in chars])