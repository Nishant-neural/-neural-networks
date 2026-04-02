import numpy as np

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
    

class CrossEntropyLoss:
    def forward(self, logits, Y):
        self.logits = logits
        self.Y = Y

        exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        self.probs = exp / np.sum(exp, axis=1, keepdims=True)

        m = logits.shape[0]

        log_likelihood = -np.log(self.probs[range(m), Y] + 1e-8)
        return np.mean(log_likelihood)

    def backward(self):
        m = self.logits.shape[0]
        grad = self.probs.copy()
        grad[range(m), self.Y] -= 1
        return grad / m
