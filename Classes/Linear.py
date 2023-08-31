import numpy as np

class Linear:
    def __init__(self, in_features, out_features, bias=True):
        self.weights = np.random.randn(in_features, out_features)
        self.biases = np.zeros((1, out_features)) # broadcasting will take care of first dimension
        self.inputs = 0
        self.dw = 0
        self.db = 0
        self.v_dw = 0
        self.v_db = 0
        
    def forward(self, x):
        self.inputs = x
        z = np.dot(x, self.weights) + self.biases
        return z
    
    def backward(self, dz):
        dz_dw = self.inputs.T
        dz_db = 1
        
        # Accumulate Gradients
        self.dw += np.dot(dz_dw, dz) / self.inputs.shape[0]
        self.db += np.sum(dz_db * dz, axis=0) / self.inputs.shape[0]
        
        # Compute da to be passed back to the previous layer
        dz_da = self.weights.T
        da = np.dot(dz, dz_da)
        return da
    
    def step(self, lr=1e-3, momentum=0):
        # Check if Momentum should be used
        if momentum <= 0:
            # Calculate Step without Momentum
            self.v_dw = self.dw
            self.v_db = self.db
        else:
            # Calculate Step with Momentum 
            self.v_dw = momentum * self.v_dw + (1 - momentum) * self.dw
            self.v_db = momentum * self.v_db + (1 - momentum) * self.db
        
        # Make Update
        self.weights -= lr * self.v_dw
        self.biases -= lr * self.v_db
    
    def zero_grad(self):
        self.dw = 0
        self.db = 0
    
    def __call__(self, x):
        return self.forward(x)