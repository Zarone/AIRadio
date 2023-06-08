from typing import Tuple
import numpy as np
import neural_networks.components.config as config

class BaseNetwork:

  """
  Parameters
  ----------
  layers 
    this defines the number of nodes in each activation layer (including input space and latent space)
  """
  def __init__(self, layers: Tuple[int]):
    self.init_coefficients(layers)

  def init_coefficients(self, layers: Tuple[int]) -> None:
    self.layers = layers
    self.biases: np.ndarray = np.empty(len(layers)-1, dtype=np.ndarray)
    self.weights: np.ndarray = np.empty(len(layers), dtype=np.ndarray)
    for i, _ in enumerate(layers[0:len(layers)-1]):
      self.biases[i] = config.rng.uniform(-1,1,(layers[i+1]))
      self.weights[i] = config.rng.uniform(-1,1,(layers[i+1], layers[i]))

  def feedforward(self, input: np.ndarray) -> np.ndarray:
    activations = np.empty(len(self.layers)-1, dtype=np.ndarray)
    # print(f"input:\n {input}")
    # print(f"weights:\n {self.weights[0]}")
    # print(f"activations:\n {activations[0]}")
    for i, _ in enumerate(activations):
      if i == 0:
        activations[i] = np.matmul(self.weights[i], input)
      else:
        print(i)
        print(activations[i-1])
        print(self.weights[i])
        activations[i] = np.matmul(self.weights[i], activations[i-1])
    return activations[-1]


  def train(self):
    pass