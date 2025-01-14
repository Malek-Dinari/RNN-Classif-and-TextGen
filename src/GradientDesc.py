class GradientDescentDemo:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update(self, weights, gradient):
        gradient = np.clip(gradient, -1, 1)
        return weights - self.learning_rate * gradient






# how to use (exple)
weights = np.random.randn(10)
gradient = np.random.randn(10) * 10
gd = GradientDescentDemo(learning_rate=0.01)
updated_weights = gd.update(weights, gradient)
print(updated_weights)