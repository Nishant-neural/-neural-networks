import numpy as np
import matplotlib.pyplot as plt

class Sequential:
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
    
    def train(self):
      for layer in self.layers:
         if hasattr(layer, "training"):
            layer.training = True

    def eval(self):
      for layer in self.layers:
         if hasattr(layer, "training"):
            layer.training = False

    def save(self, path):
      params = []
      for layer in self.layers:
         if hasattr(layer, "W"):
             params.append((layer.W, layer.b))
      np.save(path, params, allow_pickle=True)
    
    def load(self, path):
       params = np.load(path, allow_pickle=True)
       idx = 0
       for layer in self.layers:
          if hasattr(layer, "W"):
            layer.W, layer.b = params[idx]
            idx += 1
    

    def fit(self, X, Y, loss_fn, optimizer, epochs=1000, batch_size=32):

      m = X.shape[0]
      losses=[]

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
        
        losses.append(epoch_loss)
      
    
      plt.plot(losses)
      plt.title("Training Loss")
      plt.savefig("assets/loss.png")
      
    