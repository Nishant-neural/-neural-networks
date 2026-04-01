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
