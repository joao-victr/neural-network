from layer import Layer
import numpy as np

class MLP:
    def __init__(
            self,
            *layers: Layer,        
        ):
        self.layers = list(layers)

    def forward(self, input: np.ndarray):
        x = input
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad_loss: np.ndarray):
        grad = grad_loss
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def update_parameters(self, lr: float):
        for layer in self.layers:
            layer.update_parameters(lr)


