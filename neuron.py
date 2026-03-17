import numpy as np
np.random.seed(42)
r=1000
c=5
X = np.random.randn(r, c)

true_W = np.array([[2], [-1], [3], [0.5], [-2]])
true_b = -0.5


Z_true = X @ true_W + true_b
Y = (Z_true > 0).astype(int)
W = np.random.randn(c,1)
b = 0.0

def sigmoid(z):
    return 1/(1+ np.exp(-z))


learning_rate = 5
epochs=1000
for epoch in range(epochs):
     Z = X @ W + b
     A = sigmoid(Z)

     loss = -np.mean(Y*np.log(A+1e-8) + (1-Y)*np.log(1-A+1e-8))
     dZ = A - Y
     dW = X.T @ dZ / r
     db = np.sum(dZ) / r
    
     W -= learning_rate * dW
     b -= learning_rate * db

     if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")


predictions = (sigmoid(X @ W + b) > 0.5).astype(int)
accuracy = np.mean(predictions == Y)

print("Accuracy:", accuracy)



 