class Sequential:
    def __init__(self, layers):
        self.layers = layers
        
    def forward(self, x):
        curr_input = x
        for layer in self.layers:
            out = layer.forward(curr_input)
            curr_input = out
            
        return out
    
    def backward(self, dL_da):
        next_grad = dL_da
        for layer in reversed(self.layers):
            next_grad = layer.backward(next_grad)
            
    def step(self, lr=1e-3, momentum=0.9):
        for layer in reversed(self.layers):
            layer.step(lr, momentum)
    
    def zero_grad(self):
        for layer in self.layers:
            layer.zero_grad()
            
    def __call__(self, x):
        return self.forward(x)