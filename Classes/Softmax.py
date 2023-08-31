import numpy as np

class Softmax:
    def __init__(self):
        pass
    
    def softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=1).reshape(-1, 1)
    
    def forward(self, x):
        return self.softmax(x)
    
    def __call__(self, x):
        return self.forward(x)