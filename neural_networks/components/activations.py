import numpy as np
import math

def sigmoid_(x: float):
  return 1/(1+math.exp(-x))
sigmoid = np.vectorize(sigmoid_)

def relu(x: np.ndarray[float]):
  return np.maximum(0, x)