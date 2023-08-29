class Optimizer:
    def __init__(self, model, lr=1e-3, momentum=0.9):
        self.model = model
        self.lr = lr
        self.momentum = momentum
        
    def zero_grad(self):
        self.model.zero_grad()
        
    def step(self):
        self.model.step(self.lr, self.momentum)