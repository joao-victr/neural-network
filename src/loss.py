from abc import ABC, abstractmethod
import numpy as np

class LossFunction(ABC):

    @abstractmethod
    def solve(output: np.ndarray, target: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def solve_derivative(output: np.ndarray, target: np.ndarray) -> np.ndarray:
        pass


class MSE(LossFunction):

    def solve(output: np.ndarray, target: np.ndarray):
        return np.mean((output - target) ** 2)

    def solve_derivative(output: np.ndarray, target: np.ndarray):
        return 2 * (output - target) / output.size

class MAE(LossFunction):

    def solve(output: np.ndarray, target: np.ndarray):
        return np.mean(np.abs(output - target))
    
    def solve_derivative(output: np.ndarray, target: np.ndarray):
        return np.sign(output - target) / output.size

class BCE(LossFunction):
    def solve(output: np.ndarray, target: np.ndarray):
        epsilon = 1e-12
        output = np.clip(output, epsilon, 1 - epsilon)
        return -np.mean(target * np.log(output) + (1 - target) * np.log(1 - output))
        
    def solve_derivative(output: np.ndarray, target: np.ndarray):
        epsilon = 1e-12
        output = np.clip(output, epsilon, 1 - epsilon)
        return (output - target) / (output * (1 - output) * output.size)
        

