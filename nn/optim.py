import numpy as np

class SGD:
    def __init__(self, layers, lr=0.1):
        self.layers = layers
        self.lrate = lr

    def step(self):
        for layer in self.layers:
            layer.W -= self.lrate * layer.dW
            layer.b -= self.lrate * layer.db

class Adam:
    def __init__(self, layers, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.layers = layers
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0

        self.m = {}
        self.v = {}

        for idx, layer in enumerate(layers):
            self.m[idx] = {
                "W": np.zeros_like(layer.W),
                "b": np.zeros_like(layer.b)
            }
            self.v[idx] = {
                "W": np.zeros_like(layer.W),
                "b": np.zeros_like(layer.b)
            }

    def step(self):
        self.t += 1

        for idx, layer in enumerate(self.layers):

            for param_name in ["W", "b"]:

                grad = getattr(layer, "d" + param_name)

                self.m[idx][param_name] = (
                    self.beta1 * self.m[idx][param_name]
                    + (1 - self.beta1) * grad
                )

                self.v[idx][param_name] = (
                    self.beta2 * self.v[idx][param_name]
                    + (1 - self.beta2) * (grad ** 2)
                )

                m_hat = self.m[idx][param_name] / (1 - self.beta1 ** self.t)
                v_hat = self.v[idx][param_name] / (1 - self.beta2 ** self.t)

                update = self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

                setattr(layer, param_name,
                        getattr(layer, param_name) - update)
                