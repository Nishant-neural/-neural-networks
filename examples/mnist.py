import numpy as np
from sklearn.datasets import fetch_openml

from nn.layers import Linear
from nn.activations import ReLU
from nn.model import Sequential
from nn.losses import CrossEntropyLoss
from nn.optim import Adam

mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser ='liac-arff')
X = mnist.data.astype(np.float32) / 255.0
Y = mnist.target.astype(int)

X = X[:10000]
Y = Y[:10000]

X_train, X_test = X[:8000], X[8000:]
Y_train, Y_test = Y[:8000], Y[8000:]


model = Sequential([
    Linear(784, 128),
    ReLU(),
    Linear(128, 10)
])

loss_fn = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)


model.train()
model.fit(X_train, Y_train, loss_fn, optimizer,
          epochs=20,
          batch_size=64)


model.eval()
logits = model.forward(X_test)
preds = np.argmax(logits, axis=1)

accuracy = np.mean(preds == Y_test)
print("Test Accuracy:", accuracy)