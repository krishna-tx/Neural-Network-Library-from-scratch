class Sequential:
    def __init__(self, layers):
        self.layers = layers
        
    def forward(self, x):
        curr_input = x
        for layer in self.layers: # forward propagation on each layer from first to last layer
            out = layer.forward(curr_input)
            curr_input = out
            
        return out
    
    def backward(self, da):
        next_grad = da
        for layer in reversed(self.layers): # backward propagation on each layer from last to first layer
            next_grad = layer.backward(next_grad)
            
    def step(self, lr=1e-3, momentum=0.9):
        for layer in reversed(self.layers): # make a step using the calculated gradients from backpropagation
            layer.step(lr, momentum)
    
    def zero_grad(self):
        for layer in self.layers: # zero out the gradients in each layer
            layer.zero_grad()
            
    def __call__(self, x):
        return self.forward(x)