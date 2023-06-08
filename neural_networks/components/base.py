from typing import Tuple
import numpy as np
import neural_networks.components.config as config
from neural_networks.components.activations import relu, sigmoid

class BaseNetwork:

  """
  Parameters
  ----------
  layers 
    this defines the number of nodes in each activation layer (including input space and latent space)
  """
  def __init__(self, layers: Tuple[int], activation=relu):
    self.activation = activation
    self.init_coefficients(layers)

  def init_coefficients(self, layers: Tuple[int]) -> None:
    self.layers = layers
    self.biases: np.ndarray = np.empty(len(layers)-1, dtype=np.ndarray)
    self.weights: np.ndarray = np.empty(len(layers)-1, dtype=np.ndarray)
    for i in range(len(layers)-1):
      self.biases[i] = config.rng.uniform(-1,1,(layers[i+1], 1))
      self.weights[i] = config.rng.uniform(-1,1,(layers[i+1], layers[i]))

  def feedforward(self, input: np.ndarray) -> np.ndarray:
    activations = np.empty(len(self.layers)-1, dtype=np.ndarray)
    for i, _ in enumerate(activations):
      last_activations = None
      if i == 0:
        last_activations = input
      else:
        last_activations = activations[i-1]
      
      z = np.matmul(self.weights[i], last_activations) + self.biases[i]
      print(i, z)
      if i == len(activations)-1:
        activations[i] = sigmoid(z)
      else:
        activations[i] = self.activation(z)
      print(i, activations[i])
      
    return activations[-1]


  def train(self):
    pass