import numpy as np

class CrossEntropyLoss:
    def __init__(self, model):
        self.y_hat = 0
        self.y = 0
        self.model = model
        self.loss = 0
        
    def forward(self, y_hat, y):
        self.y_hat = y_hat
        self.y = y
        self.loss = np.mean(np.sum(-1 * y * np.log(y_hat + 1e-7), axis=1))
        return self
    
    def backward(self):
        # combined gradient for both Cross Entropy and Softmax
        dL_dz = self.y_hat - self.y
        self.model.backward(dL_dz)
    
    def step(self):
        pass
    
    def zero_grad(self):
        pass
    
    def item(self):
        return self.loss
    
    def __call__(self, y_hat, y):
        return self.forward(y_hat, y)
    
    def __str__(self):
        return str(self.loss)