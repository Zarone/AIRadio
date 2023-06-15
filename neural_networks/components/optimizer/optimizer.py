from abc import abstractmethod

class Optimizer:
  def __init__(self):
    pass

  @abstractmethod
  def adjusted_weight_gradient(self, weight_gradient):
    pass

  @abstractmethod
  def adjusted_bias_gradient(self, bias_gradient):
    pass
