import numpy as np
import activation

def sigmoid(x: np.ndarray):
    sig = activation.sigmoid(x)
    return sig * (1 - sig) 
