from neural_networks.components.base import BaseNetwork
from neural_networks.components.activations import\
    leaky_relu, leaky_relu_derivative
from neural_networks.components.optimizer.optimizer import Optimizer
from neural_networks.components.optimizer.adam import Adam
from typing import Tuple, List
import numpy as np
import numpy.typing as npt


class AutoEncoder(BaseNetwork):

    def __init__(
        self,
        layers: Tuple[int, ...],
        activation=leaky_relu,
        activation_derivative=leaky_relu_derivative,
        optimizer: Optimizer = Adam
    ):
        self.latent_layer = len(layers)//2
        super().__init__(
            layers,
            activation,
            activation_derivative,
            optimizer
        )

    def train(
        self,
        _training_data: npt.ArrayLike,
        max_epochs: int,
        batch_size: int = 100,
        test_data: (npt.ArrayLike | None) = None,
        learning_rate=0.05,
        graph=False,
        print_epochs=True
    ):
        training_data = np.stack((_training_data, _training_data), axis=1)
        formatted_test_data = None
        if test_data is not None:
            formatted_test_data = np.stack((test_data, test_data), axis=1)
        super().train(
            _training_data=training_data,
            max_epochs=max_epochs,
            time_separated_input=False,
            time_separated_output=False,
            batch_size=batch_size,
            test_data=formatted_test_data,
            learning_rate=learning_rate,
            graph=graph,
            print_epochs=print_epochs
        )

    def feedforward(self, input):
        return super().feedforward(input)

    def encode(self, input):
        activations: List = [None] * (self.latent_layer)
        zs: List = [None] * (self.latent_layer)

        last_activations = input
        for i in range(self.latent_layer):
            zs[i], activations[i] = self.feedforward_layer(i, last_activations)
            last_activations = activations[i]

        return (zs, activations)

    def decode(self, latent_space):
        num_layers = len(self.layers) - 1
        activations: List = [None] * (num_layers-self.latent_layer)
        zs: List = [None] * (num_layers-self.latent_layer)

        last_activations = latent_space
        for i in range(self.latent_layer, num_layers):
            zs[i-self.latent_layer], activations[i - self.latent_layer] = \
                self.feedforward_layer(i, last_activations)
            last_activations = activations[i-self.latent_layer]

        return (zs, activations)
