import numpy as np

class ReLU:
    def __init__(self):
        self.inputs = 0
    
    def forward(self, x):
        self.inputs = x
        return np.maximum(x, 0)
    
    def backward(self, da):
        da_dz = np.ones(self.inputs.shape)
        da_dz[self.inputs <= 0] = 0
    
        dz = da * da_dz
        return dz
    
    def step(self, lr=1e-3, momentum=0.9):
        pass
    
    def zero_grad(self):
        pass
    
    def __call__(self, x):
        return self.forward(x)