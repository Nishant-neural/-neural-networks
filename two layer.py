import numpy as np

# XOR dataset
X = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]])

Y = np.array([[0],
              [1],
              [1],
              [0]])

m = len(X)
np.random.seed(42)

hidden_units = 16

W1 = np.random.randn(2, hidden_units) * np.sqrt(2/2)
b1 = np.zeros((1, hidden_units))

W2 = np.random.randn(hidden_units, 1) * np.sqrt(2/hidden_units)
b2 = np.zeros((1, 1))
def relu(z):
    return np.maximum(0, z)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))
learning_rate = 0.1
epochs = 504

for epoch in range(epochs):

    # Forward
    Z1 = X @ W1 + b1
    A1 = relu(Z1)

    Z2 = A1 @ W2 + b2
    A2 = sigmoid(Z2)

    loss = -np.mean(Y*np.log(A2+1e-8) + (1-Y)*np.log(1-A2+1e-8))

    # Backward
    dZ2 = A2 - Y
    dW2 = A1.T @ dZ2 / m
    db2 = np.sum(dZ2) / m

    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * (Z1 > 0)

    dW1 = X.T @ dZ1 / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m

    # Update
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")
        predictions = (sigmoid(relu(X @ W1 + b1) @ W2 + b2) > 0.5).astype(int)

print("Predictions:")
print(predictions)
print("True labels:")
print(Y)