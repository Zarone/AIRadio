import numpy as np


def sigmoid(x: (np.ndarray | float)):
    return 1/(1+np.exp(-x))


def sigmoid_derivative(x: np.ndarray):
    sig_x = sigmoid(x)
    return sig_x * (1 - sig_x)


def relu(x: np.ndarray | float):
    return np.maximum(0, x)


def relu_derivative(x: np.ndarray):
    return (x > 0)


def linear(x: np.ndarray):
    return x


def linear_derivative(x: np.ndarray):
    return np.ones(x.shape)


def leaky_relu(x, alpha=0.01):
    return np.where(x >= 0, x, alpha * x)


def leaky_relu_derivative(x, alpha=0.01):
    return np.where(x >= 0, 1, alpha)


def elu(x, alpha=1.0):
    return np.where(x >= 0, x, alpha * (np.exp(x) - 1))


def elu_derivative(x, alpha=1.0):
    return np.where(x >= 0, 1, alpha * np.exp(x))
