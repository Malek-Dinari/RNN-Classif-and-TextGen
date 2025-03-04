import numpy as np

class GradientManager:
    @staticmethod
    def clip_grads(grads, max_norm=1.0):
        total_norm = np.sqrt(sum(np.sum(g**2) for g in grads))
        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            return [g * clip_coef for g in grads]
        return grads
    
    @staticmethod
    def xavier_init(size):
        in_dim, out_dim = size
        return np.random.randn(in_dim, out_dim) * np.sqrt(2.0 / (in_dim + out_dim))