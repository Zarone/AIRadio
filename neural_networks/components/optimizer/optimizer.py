from abc import abstractmethod
import numpy as np


class Optimizer:
    EPSILON_NAUGHT = 1E-15
    SQRT = np.vectorize(
        lambda x: np.sqrt(x), otypes=[np.ndarray]
    )

    def __init__(self):
        pass

    @abstractmethod
    def adjusted_gradient(self, vector_id, gradient, *args):
        """
        :param v_id a unique identifier for that particular type of gradient vector.
Ex: 0 to represent biases, 1 to represent weights
        """
        pass
