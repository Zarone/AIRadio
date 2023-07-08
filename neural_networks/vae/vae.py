from neural_networks.components.base import BaseNetwork
from neural_networks.components.activations import (
    leaky_relu,
    leaky_relu_derivative
)
from neural_networks.components.optimizer.optimizer import Optimizer
from neural_networks.components.optimizer.adam_w_taperoff import AdamTaperoff
import numpy as np
from typing import Tuple


class VAE(BaseNetwork):
    def __init__(
        self,
        encoder_layers: Tuple[int, ...],
        decoder_layers: Tuple[int, ...],
        activation=leaky_relu,
        activation_derivative=leaky_relu_derivative,
        optimizer: Optimizer = AdamTaperoff(),
        sub_network: BaseNetwork = BaseNetwork
    ) -> None:
        assert encoder_layers[-1] == decoder_layers[0],\
            "Initialized VAE with inconsistent latent space size"

        self.optimizer = optimizer
        self.activation = activation
        self.activation_derivative = activation_derivative

        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers

        # This is because the last layer of the decoder
        # has to contain both the mean and the log variance
        e_layers = list(encoder_layers)
        e_layers[-1] *= 2

        self.encoder = sub_network(e_layers)
        self.decoder = sub_network(decoder_layers)

        self.init_coefficients()

    def init_coefficients(self):
        self.encoder.init_coefficients()
        self.decoder.init_coefficients()

    def encode(self, input):
        return self.encoder.feedforward(input)

    def gen(self, mu, log_variance) -> Tuple[np.ndarray, np.ndarray]:
        epsilon = np.random.randn(len(mu)).reshape(-1, 1)
        z = mu + np.exp(0.5 * log_variance) * epsilon
        return z

    def decode(self, generated):
        return self.decoder.feedforward(generated)

    def feedforward(self, input):
        encoded = self.encode(input)
        generated = self.get(encoded)
        decoded = self.decode(generated)
        return decoded

    def train(self, training_data, *args):
        return super().train(
            self.format_unsupervised_input(training_data), *args
        )

