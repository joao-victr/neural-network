import numpy as np

def reLU(x: np.ndarray):
    return np.maximum(0, x)

def sigmoid(x: np.ndarray):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x: np.ndarray):
    sig = sigmoid(x)
    return sig * (1 - sig) 

def softmax(x: np.ndarray):
    return np.exp(x - max(x)) / np.sum(np.exp(x - np.max(x)))

