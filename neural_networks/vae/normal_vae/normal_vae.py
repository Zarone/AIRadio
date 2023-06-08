from neural_networks.components.base import BaseNetwork
import numpy as np

class VAE(BaseNetwork):
  
  def encode(self, input: np.ndarray) -> np.ndarray:
    activations = np.empty(
      (len(self.layers)-1)//2, 
      dtype=np.ndarray
    )
    for i, _ in enumerate(activations):
      last_activations = None
      if i == 0:
        last_activations = input
      else:
        last_activations = activations[i-1]
      z = np.matmul(self.weights[i], last_activations) + self.biases[i]
      activations[i] = self.activation(z)
    return activations[-1]
