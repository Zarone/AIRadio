from neural_networks.components.optimizer.optimizer import Optimizer


class Momentum(Optimizer):
  def __init__(self, Beta: float):
    self.Beta = Beta
    self.v_dw = None
    self.v_db = None

  def adjusted_weight_gradient(self, weight_gradient):
    return Beta * v_dw + (1-Beta) * weight_gradient
    pass

  def adjusted_bias_gradient(self, bias_gradient):
    pass
