from neural_networks.components.base import BaseNetwork
import numpy as np
from neural_networks.components.activations import relu
from typing import Tuple, Callable

class VAE(BaseNetwork):
  
  def __init__(
      self, 
      layers: Tuple[int, ...], 
      activation=relu, 
      activation_exceptions: dict[int, Callable]={}
  ) -> None:
    if (len(layers) % 2 == 0): raise Exception("Initialized VAE with odd number of layers")
    super().__init__(layers, activation, activation_exceptions)


  def encode(self, input: np.ndarray) -> np.ndarray:
    activations = np.empty(
      (len(self.layers)-1)//2, 
      dtype=np.ndarray
    )
    for i in range(0, len(activations)):
      last_activations = input if i==0 else activations[i-1]
      activations[i] = self.feedforward_layer(i, last_activations)
    return activations[-1]


  def decode(self, input: np.ndarray) -> np.ndarray:
    activations = np.empty(
      (len(self.layers)-1)//2, 
      dtype=np.ndarray
    )
    for i in range(len(activations), len(self.layers)-1):
      last_activations = input if i==len(activations) else activations[i-1-len(activations)]
      activations[i-len(activations)] = self.feedforward_layer(i, last_activations)
    return activations[-1]
