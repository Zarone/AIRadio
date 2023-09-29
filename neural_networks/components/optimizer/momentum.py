from neural_networks.components.optimizer.optimizer import Optimizer
import numpy as np


class Momentum(Optimizer):
    def __init__(
        self,
        Beta: float = 0.95,
    ):
        self.v_vector = dict()

        self.Beta = Beta

    def adjusted_gradient(self, v_id, vector, *_):
        if self.v_vector.get(v_id, None) is None:
            self.v_vector[v_id] = np.empty((vector.shape), dtype=np.ndarray)
            for index, _ in enumerate(vector):
                self.v_vector[v_id][index] = np.zeros(vector[index].shape)

        self.v_vector[v_id] = self.Beta * self.v_vector[v_id] + \
            (1-self.Beta) * vector

        return self.v_vector[v_id]
