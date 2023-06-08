import numpy as np
import math

def relu_(x: float):
  if (x > 0):
    return x
  else:
    return 0.0
relu_element_vectorized = np.vectorize(relu_)

def sigmoid_(x: float):
  return 1/(1+math.exp(-x))
sigmoid_element_vectorized = np.vectorize(sigmoid_)

def relu(x: np.ndarray[float]):
  return relu_element_vectorized(x)

def sigmoid(x: np.ndarray[float]):
  return sigmoid_element_vectorized(x)
