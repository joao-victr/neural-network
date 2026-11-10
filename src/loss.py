import numpy as np


def mse(output: np.ndarray, target: np.ndarray):
    return np.mean((output - target) ** 2)


def mse_derivative(output: np.ndarray, target: np.ndarray):
    return 2 * (output - target) / output.size


def mae(output: np.ndarray, target: np.ndarray):
    return np.mean(np.abs(output - target))


def mae_derivative(output: np.ndarray, target: np.ndarray):
    return np.sign(output - target) / output.size


def bce(output: np.ndarray, target: np.ndarray):
    epsilon = 1e-12
    output = np.clip(output, epsilon, 1 - epsilon)
    return -np.mean(target * np.log(output) + (1 - target) * np.log(1 - output))


def bce_derivative(output: np.ndarray, target: np.ndarray):
    epsilon = 1e-12
    output = np.clip(output, epsilon, 1 - epsilon)
    return (output - target) / (output * (1 - output) * output.size)
