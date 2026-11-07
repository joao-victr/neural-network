import numpy as np
from src.activation import reLU
from typing import Literal, Callable

class Layer:
    def __init__(
            self, 
            n_input: int, 
            n_output: int, 
            init_method : Literal['he', 'xavier'], 
            activation: Callable[[np.ndarray], np.ndarray]
        ):
        self.n_output = n_output
        self.n_input = n_input
        self.weights = np.zeros((n_output, n_input))
        self.bias = np.zeros((n_output))
        self.init_method = init_method
        self.activation = activation


    def forward(self, inputs: np.ndarray):
        if len(inputs) != self.n_input:
            raise("Input dimension must match the number of inputs")

        output = np.dot(self.weights, inputs) + self.bias
        return self.activation(output) 

    def initialize_weights(self):
        if self.init_method == 'xavier':
            std = np.sqrt(2 / (self.n_input + self.n_output))
            self.weights = np.random.randn(self.n_output, self.n_input) * std
        elif self.init_method == 'he':
            std = np.sqrt(2 / self.n_input)
            self.weights = np.random.randn(self.n_output, self.n_input) * std




