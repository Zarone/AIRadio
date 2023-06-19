from numpy import number
from neural_networks.components.base import BaseNetwork
from neural_networks.components.activations import relu
from typing import Callable, Tuple

class AutoEncoder(BaseNetwork):
  def __init__(
      self, 
      layers: Tuple[int, ...], 
      latent_layer: number,
      activation=relu, 
      activation_exceptions: dict[int, Callable]={}
  ) -> None:
    self.activation = activation
    self.activation_exceptions = activation_exceptions
    self.latent_layer = latent_layer
    self.init_coefficients(layers)



