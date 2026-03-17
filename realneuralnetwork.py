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
    
class ReLU:
    def forward(self, X):
        self.X = X
        return np.maximum(0, X)

    def backward(self, dA):
        return dA * (self.X > 0)
    

class Sigmoid:
    def forward(self, X):
        self.A = 1 / (1 + np.exp(-X))
        return self.A

    def backward(self, dA):
        return dA * self.A * (1 - self.A)
    
    
class sequential:
    def __init__(self,layers):
       self.layers = layers

    def forward(self,X):
        for layer in self.layers:
             X = layer.forward(X)
        return X
    
    def backward(self, grad):
      for layer in reversed(self.layers):
        grad = layer.backward(grad)
    
    def parameters(self):
        params = []
        for layer in self.layers:
            if hasattr(layer, "W"):
                params.append(layer)
        return params
    
    
    def fit(self, X, Y, loss_fn, optimizer, epochs=1000, batch_size=32):

      m = X.shape[0]

      for epoch in range(epochs):

        indices = np.random.permutation(m)
        X_shuffled = X[indices]
        Y_shuffled = Y[indices]

        epoch_loss = 0

        for i in range(0, m, batch_size):

            X_batch = X_shuffled[i:i+batch_size]
            Y_batch = Y_shuffled[i:i+batch_size]

            preds = self.forward(X_batch)

            loss = loss_fn.forward(preds, Y_batch)
            epoch_loss += loss

            grad = loss_fn.backward()
            self.backward(grad)

            optimizer.step()

        if epoch % 200 == 0:
            print(f"Epoch {epoch}, Loss: {epoch_loss}")
    
     


class BCELoss:
    def forward(self, Y_pred, Y_true):
        self.Y_pred = Y_pred
        self.Y_true = Y_true
        return -np.mean(Y_true*np.log(Y_pred+1e-8) +
            (1-Y_true)*np.log(1-Y_pred+1e-8)
        )

    def backward(self):
      return  -(self.Y_true / (self.Y_pred + 1e-8)) + \
          (1 - self.Y_true) / (1 - self.Y_pred + 1e-8)


class SGD:
    def __init__(self, layers, lr=0.1):
        self.layers = layers
        self.lrate = lr

    def step(self):
        for layer in self.layers:
            layer.W -= self.lrate * layer.dW
            layer.b -= self.lrate * layer.db


X = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]])

Y = np.array([[0],
              [1],
              [1],
              [0]])

model = sequential([
    Linear(2, 16),
    ReLU(),
    Linear(16, 1),
    Sigmoid()
])

loss_fn = BCELoss()
optimizer = SGD(model.parameters(), lr=0.1)
model.fit(X, Y, loss_fn, optimizer,
          epochs=1000,
          batch_size=2)


predictions = (model.forward(X) > 0.5).astype(int)

print("Final Predictions:")
print(predictions)
print("True labels:")
print(Y)
