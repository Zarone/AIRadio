from neural_networks.components.optimizer.optimizer import Optimizer
import numpy as np

class Momentum(Optimizer):
  def __init__(self, Beta: float = 0.5):
    self.Beta = Beta
    self.v_dw = None
    self.v_db = None

  def adjusted_weight_gradient(self, weight_gradient):
    if self.v_dw is None:
      self.v_dw = np.empty(weight_gradient.shape, dtype=np.ndarray)
      for index, _ in enumerate(weight_gradient):
        self.v_dw[index] = np.zeros(weight_gradient[index].shape)

    self.v_dw = self.Beta * self.v_dw + (1-self.Beta) * weight_gradient

    return self.v_dw

  def adjusted_bias_gradient(self, bias_gradient):
    if self.v_db is None:
      self.v_db = np.empty(bias_gradient.shape, dtype=np.ndarray)
      for index, _ in enumerate(bias_gradient):
        self.v_db[index] = np.zeros(bias_gradient[index].shape)

    self.v_db = self.Beta * self.v_db + (1-self.Beta) * bias_gradient 

    return self.v_db
