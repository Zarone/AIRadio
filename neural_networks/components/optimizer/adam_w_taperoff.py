from neural_networks.components.optimizer.optimizer import Optimizer
import numpy as np


class AdamTaperoff(Optimizer):
    def __init__(
        self,
        Beta1: float = 0.95,
        Beta2: float = 0.95,
    ):
        self.v_vector = dict()
        self.s_vector = dict()

        self.Beta1 = Beta1
        self.Beta2 = Beta2

    def adjusted_gradient(self, v_id, vector, loss, *_):
        if self.v_vector.get(v_id, None) is None:
            self.v_vector[v_id] = np.empty((vector.shape), dtype=np.ndarray)
            for index, _ in enumerate(vector):
                self.v_vector[v_id][index] = np.zeros(vector[index].shape)

        if self.s_vector.get(v_id, None) is None:
            self.s_vector[v_id] = np.empty((vector.shape), dtype=np.ndarray)
            for index, _ in enumerate(vector):
                self.s_vector[v_id][index] = np.zeros(vector[index].shape)

        self.v_vector[v_id] = self.Beta1 * self.v_vector[v_id] + \
            (1-self.Beta1) * vector
        self.s_vector[v_id] = self.Beta2 * self.s_vector[v_id] + \
            (1-self.Beta2) * np.square(vector)

        taper_off = min(loss, 1)

        # Multiplying by the square root of loss is just
        # so that we can get precision for small data sets.
        # I haven't tested it much, but it seems pretty cool.
        return self.v_vector[v_id] * Optimizer.SQRT(
            taper_off / (self.s_vector[v_id] + Optimizer.EPSILON_NAUGHT)
        )
