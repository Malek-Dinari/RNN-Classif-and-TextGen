import abc
import numpy as np

class BaseRNN(abc.ABC):
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size

    @abc.abstractmethod
    def forward(self, x, h_prev):
        pass

    @abc.abstractmethod
    def backward(self, dy):
        pass

    @abc.abstractmethod
    def init_hidden(self, batch_size):
        pass

    @abc.abstractmethod
    def zero_grad(self):
        pass