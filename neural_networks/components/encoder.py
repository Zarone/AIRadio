from neural_networks.components.base import BaseNetwork
from typing import Tuple
import numpy as np
from sklearn.model_selection import train_test_split

class Encoder(BaseNetwork):


  def __init__(self, layers: Tuple[int]):
    self.init_coefficients(layers)