from neural_networks.components.base import BaseNetwork
from neural_networks.components.activations import (
    leaky_relu,
    leaky_relu_derivative
)
from neural_networks.components.optimizer.optimizer import Optimizer
from neural_networks.components.optimizer.adam_w_taperoff import AdamTaperoff
from neural_networks.components.loss_types import Loss
from neural_networks.components.coefs import Coefficients
from neural_networks.components.feedforward_data import FeedforwardData
import numpy as np
from typing import Tuple


class VAE(BaseNetwork):
    def __init__(
        self,
        encoder_layers: Tuple[int, ...],
        decoder_layers: Tuple[int, ...],
        activation=leaky_relu,
        activation_derivative=leaky_relu_derivative,
        optimizer: Optimizer = AdamTaperoff,
        sub_network: BaseNetwork = BaseNetwork
    ) -> None:
        assert encoder_layers[-1] == decoder_layers[0],\
            "Initialized VAE with inconsistent latent space size"

        self.optimizer = optimizer()
        self.activation = activation
        self.activation_derivative = activation_derivative

        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers

        # This is because the last layer of the decoder
        # has to contain both the mean and the log variance
        e_layers = list(encoder_layers)
        e_layers[-1] *= 2

        self.layers = encoder_layers[:-1] + decoder_layers

        self.latent_size = self.decoder_layers[0]

        self.encoder: BaseNetwork = sub_network(e_layers)
        self.decoder: BaseNetwork = sub_network(decoder_layers)

        self.init_coefficients()
        self.beta = 1
        self.epsilon_max = 0

    def init_coefficients(self):
        self.encoder.init_coefficients()
        self.decoder.init_coefficients()

    def kl_loss(
        self,
        mu,
        log_var
    ):
        # Regularization term, KL divergence
        kl_loss = -0.5 * np.sum(
            1 + log_var - np.square(mu) - np.exp(log_var)
        ) / len(mu)

        return {
            Loss.KL_LOSS: kl_loss
        }

    def get_time_seperated_data(self, input_data):
        return self.encoder.get_time_seperated_data(input_data)

    def _encode(self, input, num_inputs):
        encoder_values = self.encoder._feedforward(
            input,
            num_inputs=num_inputs,
            num_outputs=1
        )

        # Get the output, then the last time step,
        # then the activations, and then the last
        # layer of activations
        last_layer = encoder_values[FeedforwardData.OUTPUT][-1][1][-1]
        parameters_count = len(last_layer)//2
        mu = last_layer[:parameters_count]
        log_var = last_layer[parameters_count:]
        return (mu, log_var, encoder_values)

    def encode(self, input):
        last_layer = self.encoder.feedforward(input)
        parameters_count = len(last_layer)//2
        mu = last_layer[:parameters_count]
        log_var = last_layer[parameters_count:]
        return (mu, log_var)

    def _gen(self, mu, log_variance) -> Tuple[np.ndarray, np.ndarray]:
        epsilon = self.epsilon_max * np.random.randn(len(mu)).reshape(-1, 1)
        z = mu + np.exp(0.5 * log_variance) * epsilon
        return (z, epsilon)

    def gen(self, mu, log_variance):
        return self._gen(mu, log_variance)[0]

    def decode(self, generated):
        return self.decoder.feedforward(generated)

    def feedforward(self, input):
        mu, log_var = self.encode(input)
        generated = self.gen(mu, log_var)
        decoded = self.decode(generated)
        return decoded

    def train(self, training_data, *args, **kwargs):
        return super().train(
            self.format_unsupervised_input(training_data), *args, **kwargs
        )

    def init_gradients(self):
        self.encoder.init_gradients()
        self.decoder.init_gradients()

    def gen_backprop(self, dL_dz, mu, log_variance, epsilon):
        # This calculation is derived by taking the partial
        # derivative of loss with respect to mu.
        dL_dmu = dL_dz + self.beta*mu/self.latent_size

        # This calculation is derived by taking the partial
        # derivative of loss with respect to logvar.
        dL_dlogvar = \
            dL_dz \
            * epsilon/2*np.exp(log_variance/2) \
            - self.beta*(1-np.exp(log_variance))/self.latent_size/2

        return np.concatenate((dL_dmu, dL_dlogvar), axis=0)

    def update_coefficients(self, learning_rate, batch, losses):
        self.encoder.update_coefficients(learning_rate, batch, losses)
        self.decoder.update_coefficients(learning_rate, batch, losses)

    def backpropagate(
        self,
        data_point,
        print_epochs,
        time_separated_values: bool,
        dL_dz=None
    ):
        reconstruction_loss = 0
        kl_loss = 0

        num_inputs = len(data_point) if time_separated_values else 1
        mu, log_var, encoder_values = self._encode(
            data_point, num_inputs=num_inputs
        )

        generated, epsilon = self._gen(mu, log_var)

        decoder_loss, dL_dz = self.decoder.backpropagate(
            np.array([generated, data_point[1]], dtype=np.ndarray),
            print_epochs
        )

        dL_dz = self.gen_backprop(dL_dz, mu, log_var, epsilon)

        reconstruction_loss += decoder_loss[Loss.RECONSTRUCTION_LOSS]
        kl_loss += self.kl_loss(mu, log_var)[Loss.KL_LOSS]

        encoder_loss, _ = self.encoder.backpropagate(
            [data_point[0]],
            False,
            dL_dz=dL_dz,
            _feedforward_values=encoder_values
        )

        return ({
            Loss.RECONSTRUCTION_LOSS: reconstruction_loss,
            Loss.KL_LOSS: kl_loss,
            Loss.TOTAL_LOSS: reconstruction_loss + kl_loss
        }, None)
