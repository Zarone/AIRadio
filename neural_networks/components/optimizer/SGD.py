from neural_networks.components.optimizer.optimizer import Optimizer


class SGD(Optimizer):
    def adjusted_gradient(self, v_id, gradient, *_):
        return gradient
