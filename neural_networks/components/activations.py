import numpy as np
import math

def sigmoid_(x: float) -> float:
  return 1/(1+math.exp(-x))
sigmoid = np.vectorize(sigmoid_)

def sigmoid_derivative(x: np.ndarray):
  sig_x = sigmoid(x)
  return sig_x * (1 - sig_x)

def relu(x: np.ndarray):
  return np.maximum(0, x)

def relu_derivative(x: np.ndarray):
  return (x > 0) * 1.0

def linear(x: np.ndarray):
  return x

def linear_derivative(x: np.ndarray):
  return np.ones(x.shape)