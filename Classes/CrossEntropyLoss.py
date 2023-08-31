import numpy as np

class CrossEntropyLoss:
    def __init__(self, model):
        self.y_hat = 0
        self.y = 0
        self.model = model
        self.loss = 0
        
    def softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=1).reshape(-1, 1)
        
    def forward(self, x, y):
        self.y_hat = self.softmax(x) # do a softmax to get predictions
        self.y = y
        self.loss = np.mean(np.sum(-1 * y * np.log(self.y_hat + 1e-7), axis=1))
        return self
    
    def backward(self):
        dz = self.y_hat - self.y # Combined partial derivative for both Cross Entropy and Softmax
        self.model.backward(dz)
    
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