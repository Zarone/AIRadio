from neural_networks.components.optimizer.optimizer import Optimizer
import numpy as np
import math

class Adam(Optimizer):
    EPSILON_NAUGHT = 1E-15
    SQRT = np.vectorize(
        lambda x: np.sqrt(x), otypes=[np.ndarray])

    def __init__(
        self,
        Beta1: float = 0.95,
        Beta2: float = 0.95,
        loss_taperoff=False
    ):
        self.v_dw = None
        self.v_db = None
        self.s_dw = None
        self.s_db = None
        self.Beta1 = Beta1
        self.Beta2 = Beta2
        self.experiental_loss_taperoff = loss_taperoff

    def adjusted_weight_gradient(self, weight_gradient, loss):
        if self.v_dw is None:
            self.v_dw = np.empty(weight_gradient.shape, dtype=np.ndarray)
            for index, _ in enumerate(weight_gradient):
                self.v_dw[index] = np.zeros(weight_gradient[index].shape)

        if self.s_dw is None:
            self.s_dw = np.empty(weight_gradient.shape, dtype=np.ndarray)
            for index, _ in enumerate(weight_gradient):
                self.s_dw[index] = np.zeros(weight_gradient[index].shape)

        self.v_dw = self.Beta1 * self.v_dw + (1-self.Beta1) * weight_gradient
        self.s_dw = self.Beta2 * self.s_dw + \
            (1-self.Beta2) * np.square(weight_gradient)

        taper_off = min(loss, 1)\
            if self.experiental_loss_taperoff else 1

        # Multiplying by the square root of loss is just
        # so that we can get precision for small data sets.
        # I haven't tested it much, but it seems pretty cool.
        return self.v_dw * Adam.SQRT(
            taper_off / (self.s_dw + Adam.EPSILON_NAUGHT)
        )

    def adjusted_bias_gradient(self, bias_gradient, loss):
        if self.v_db is None:
            self.v_db = np.empty(bias_gradient.shape, dtype=np.ndarray)
            for index, _ in enumerate(bias_gradient):
                self.v_db[index] = np.zeros(bias_gradient[index].shape)

        if self.s_db is None:
            self.s_db = np.empty(bias_gradient.shape, dtype=np.ndarray)
            for index, _ in enumerate(bias_gradient):
                self.s_db[index] = np.zeros(bias_gradient[index].shape)

        self.v_db = self.Beta1 * self.v_db + (1-self.Beta1) * bias_gradient
        self.s_db = self.Beta2 * self.s_db + \
            (1-self.Beta2) * np.square(bias_gradient)

        taper_off = min(loss, 1)\
            if self.experiental_loss_taperoff else 1

        return self.v_db * Adam.SQRT(
            taper_off / (self.s_db + Adam.EPSILON_NAUGHT)
        )
