import numpy as np

from nn.layers import Linear, Dropout
from nn.activations import ReLU, Sigmoid
from nn.model import Sequential
from nn.losses import BCELoss
from nn.optim import Adam

X = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]])

Y = np.array([[0],
              [1],
              [1],
              [0]])

model = Sequential([
    Linear(2, 16),
    ReLU(),
    Dropout(0.2),
    Linear(16, 1),
    Sigmoid()
])

loss_fn = BCELoss()
optimizer = Adam(model.parameters(), lr=0.01)

model.train()
model.fit(X, Y, loss_fn, optimizer, epochs=1000, batch_size=2)
model.eval()

predictions = (model.forward(X) > 0.5).astype(int)

print("Predictions:", predictions)