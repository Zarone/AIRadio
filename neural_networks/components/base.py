from typing import Tuple, Callable
import numpy as np
import neural_networks.components.config as config
from neural_networks.components.activations import relu
import math
import matplotlib.pyplot as plt

class BaseNetwork:

  """
  Parameters
  ----------
  layers 
    This defines the number of nodes in each activation layer (including input space and latent space)
  activation
    This is the primary activation function which the neural network uses
  activation_exceptions
    This is a dictionary where the key equals the layer where the exception occurs, and the key is the replacement activation function
  """
  def __init__(
      self, 
      layers: Tuple[int, ...], 
      activation=relu, 
      activation_exceptions: dict[int, Callable]={}
  ) -> None:
    self.activation = activation
    self.activation_exceptions = activation_exceptions
    self.init_coefficients(layers)

  def init_coefficients(self, layers: Tuple[int]) -> None:
    self.layers = layers
    self.biases: np.ndarray = np.empty(len(layers)-1, dtype=np.ndarray)
    self.weights: np.ndarray = np.empty(len(layers)-1, dtype=np.ndarray)
    for i in range(len(layers)-1):
      max = math.sqrt(2/layers[i])
      min = -max
      self.biases[i] = config.rng.uniform(min,max,(layers[i+1], 1))
      self.weights[i] = config.rng.uniform(min,max,(layers[i+1], layers[i]))

  def feedforward(self, input: np.ndarray) -> np.ndarray:
    activations = np.empty(len(self.layers)-1, dtype=np.ndarray)
    for i, _ in enumerate(activations):
      last_activations = input if i==0 else activations[i-1]
      _, activations[i] = self.feedforward_layer(i, last_activations)
    return activations[-1]

  def feedforward_layer(self, i: int, last_activations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # z_{i} = w * a_{i-1} + b
    z = np.matmul(self.weights[i], last_activations) + self.biases[i]

    # Sometimes, the default activation function, self.activation,
    # will not always be the activation for every layer. For instance,
    # ReLu is not often the activation for the final layer since negative
    # results could not then be returned ever, so self.activation_exceptions
    # would have a value at key=final_layer_index with value=sigmoid
    activation_function = self.activation_exceptions.get(i, self.activation)
    return (z, activation_function(z))
