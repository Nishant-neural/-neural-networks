import numpy as np

class Linear:
    def __init__(self, in_features, out_features):
        self.W = np.random.randn(in_features, out_features) * np.sqrt(2/in_features)
        self.b = np.zeros((1, out_features))

    def forward(self, X):
        self.X = X       
        return X @ self.W + self.b
    def backward(self, dZ):
        m = self.X.shape[0]

        self.dW = self.X.T @ dZ / m
        self.db = np.sum(dZ, axis=0, keepdims=True) / m

        return dZ @ self.W.T
    


class Dropout:
    def __init__(self, p=0.5):
        self.p = p
        self.training = True

    def forward(self, X):
        if self.training:
            self.mask = (np.random.rand(*X.shape) > self.p) / (1 - self.p)
            return X * self.mask
        else:
            return X

    def backward(self, dA):
        return dA * self.mask
