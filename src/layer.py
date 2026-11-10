import numpy as np
from src.activation import reLU
from typing import Literal, Callable

class Layer:
    def __init__(
            self, 
            n_input: int, 
            n_output: int, 
            init_method : Literal['he', 'xavier'], 
            activation_function: Callable[[np.ndarray], np.ndarray],
            activation_function_derivative: Callable[[np.ndarray], np.ndarray],
            learning_rate: float
        ):
        self.n_output = n_output
        self.n_input = n_input
        self.weights = np.zeros((n_output, n_input))
        self.bias = np.zeros((n_output))
        self.init_method = init_method
        self.activation_function = activation_function
        self.activation_function_derivative = activation_function_derivative
        self.learning_rate = learning_rate

        self.input: np.ndarray
        self.activation_value: np.ndarray
        self.pre_act_output: np.ndarray
        self.delta: np.ndarray
        self.gradients: np.ndarray

    def forward(self, inputs: np.ndarray):
        if len(inputs) != self.n_input:
            raise("Input dimension must match the number of inputs")
        self.input = inputs
        self.pre_act_output = np.dot(self.weights, inputs) + self.bias
        self.activation_value = self.activation_function(self.pre_act_output)
        return self.activation_value

    def initialize_weights(self):
        if self.init_method == 'xavier':
            std = np.sqrt(2 / (self.n_input + self.n_output))
            self.weights = np.random.randn(self.n_output, self.n_input) * std
        elif self.init_method == 'he':
            std = np.sqrt(2 / self.n_input)
            self.weights = np.random.randn(self.n_output, self.n_input) * std

    def backward(self, next_delta, weights):
        self.delta = np.dot(weights.T, next_delta) * self.activation_function_derivative(self.pre_act_output)
        self.gradients = np.outer(self.delta, self.input)
        return self.delta

    def backward_output(self, loss_derivative: Callable[[np.ndarray], np.ndarray]):
        d_loss = loss_derivative(self.activation_value)
        self.delta = d_loss * self.activation_function_derivative(self.pre_act_output)
        self.gradients = np.outer(self.delta, self.input)
        return self.delta

    def update_parameters(self):
        self.weights = self.weights - self.learning_rate * self.gradients
        self.bias = self.bias - self.learning_rate * self.delta
