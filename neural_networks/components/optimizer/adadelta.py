from neural_networks.components.optimizer.optimizer import Optimizer
import numpy as np


class Adadelta(Optimizer):
    def __init__(self, Beta: float = 0.95):
        self.s_vector = dict()
        self.Beta = Beta

    def adjusted_gradient(self, v_id, vector, *_):
        if self.s_vector.get(v_id, None) is None:
            self.s_vector[v_id] = np.empty((vector.shape), dtype=np.ndarray)
            for index, _ in enumerate(vector):
                self.s_vector[v_id][index] = np.zeros(vector[index].shape)

        self.s_vector[v_id] = self.Beta * self.s_vector[v_id] + \
            (1-self.Beta) * np.square(vector)

        return vector / Optimizer.SQRT(
            (self.s_vector[v_id] + Optimizer.EPSILON_NAUGHT)
        )
