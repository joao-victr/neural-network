from abc import ABC, abstractmethod
import numpy as np

class ActivationFunction(ABC):

    @abstractmethod
    def solve(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def solve_derivative(self, x: np.ndarray) -> np.ndarray:
        pass


class ReLU(ActivationFunction):

    def solve(self, x: np.ndarray):
        return np.maximum(0, x)    

    def solve_derivative(self, x: np.ndarray):
        return (x > 0).astype(float)

class Sigmoid(ActivationFunction):

    def solve(self, x: np.ndarray):
        return 1 / (1 + np.exp(-x))
    
    def solve_derivative(self, x: np.ndarray):
        sig = self.solve(x)
        return sig * (1 - sig)         

class Softmax(ActivationFunction):

    def solve(self, x: np.ndarray):
        return np.exp(x - max(x)) / np.sum(np.exp(x - np.max(x)))

