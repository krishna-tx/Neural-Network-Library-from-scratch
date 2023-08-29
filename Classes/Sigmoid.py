import numpy as np

class Sigmoid:
    def __init__(self):
        self.inputs = 0
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def forward(self, x):
        self.inputs = x
        return self.sigmoid(x)
    
    def backward(self, da):
        da_dz = self.sigmoid(self.inputs) * (1 - self.sigmoid(self.inputs))
        dz = da * da_dz
        return dz
    
    def step(self, lr=1e-3, momentum=0.9):
        pass
    
    def zero_grad(self):
        pass
    
    def __call__(self, x):
        return self.forward(x)