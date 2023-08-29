import numpy as np

class Softmax:
    def __init__(self):
        self.inputs = 0
        
    def softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=1).reshape(-1, 1)
    
    def forward(self, x):
        self.inputs = x
        return self.softmax(x)
    
    def backward(self, dz):
        # combined gradient is calculated from Cross Entropy Loss
        return dz
    
    def step(self, lr=1e-3, momentum=0.9):
        pass
    
    def zero_grad(self):
        pass
    
    def __call__(self, x):
        return self.forward(x)