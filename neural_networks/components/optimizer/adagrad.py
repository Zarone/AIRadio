from neural_networks.components.optimizer.optimizer import Optimizer
import numpy as np

class Adagrad(Optimizer):
  def __init__(self):
    self.s_dw = None
    self.s_db = None

  def adjusted_weight_gradient(self, weight_gradient):
    if self.s_dw is None:
      self.s_dw = np.empty(weight_gradient.shape, dtype=np.ndarray)
      for index, _ in enumerate(weight_gradient):
        self.s_dw[index] = np.zeros(weight_gradient[index].shape)

    self.s_dw += np.square(weight_gradient)

    # This is somehow faster than vectorization
    for index, _ in enumerate(weight_gradient):
      weight_gradient[index] /= np.sqrt(self.s_dw[index] + 0.0001)

    return weight_gradient

  def adjusted_bias_gradient(self, bias_gradient):
    if self.s_db is None:
      self.s_db = np.empty(bias_gradient.shape, dtype=np.ndarray)
      for index, _ in enumerate(bias_gradient):
        self.s_db[index] = np.zeros(bias_gradient[index].shape)

    self.s_db += np.square(bias_gradient)

    for index, _ in enumerate(bias_gradient):
      bias_gradient[index] /= np.sqrt(self.s_db[index] + 0.0001)

    return bias_gradient
