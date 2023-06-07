from typing import Tuple
import numpy as np
import neural_networks.components.config as config

class BaseNetwork:
  

  def init_coefficients(self, layers: Tuple[int]):
    self.biases = np.empty(len(layers)-1, dtype=np.ndarray)
    self.weights = np.empty(len(layers), dtype=np.ndarray)
    for i, _ in enumerate(layers[0:len(layers)-1]):
      self.biases[i] = config.rng.uniform(-1,1,(layers[i+1]))
      self.weights[i] = config.rng.uniform(-1,1,(layers[i], layers[i+1]))