from neural_networks.components.optimizer.optimizer import Optimizer

class SGD(Optimizer):
  def adjusted_bias_gradient(self, bias_gradient):
    return bias_gradient 

  def adjusted_weight_gradient(self, weight_gradient):
    return weight_gradient

