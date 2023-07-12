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
        encoder_args: dict,
        decoder_args: dict,
        latent_size: int,
        activation=leaky_relu,
        activation_derivative=leaky_relu_derivative,
        optimizer: Optimizer = AdamTaperoff,
        sub_network: BaseNetwork = BaseNetwork
    ) -> None:
        self.optimizer = optimizer()
        self.activation = activation
        self.activation_derivative = activation_derivative

        self.latent_size = latent_size

        self.encoder: BaseNetwork = sub_network(**encoder_args)
        self.decoder: BaseNetwork = sub_network(**decoder_args)

        # This is just for better error handling, specifically
        # if the two initial arguments for the encoder and
        # decoder aren't compatible.
        sub_network.check_vae_compatibility(self.encoder, self.decoder)

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

    def _encode(self, input, time_separated_inputs: bool):
        num_inputs = len(input) if time_separated_inputs else 1

        encoder_values = self.encoder._feedforward(
            input,
            time_separated_inputs=time_separated_inputs,
            iterations=num_inputs
        )

        last_layer = None
        if time_separated_inputs:
            # Get the output, then the last time step,
            # then the activations, and then the last
            # layer of activations.
            last_layer = encoder_values[FeedforwardData.OUTPUT][-1][1][-1]
        else:
            # Get the output, then the activations, and
            # then the last layer of the activations.
            last_layer = encoder_values[FeedforwardData.OUTPUT][1][-1]

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
        time_separated: bool,
        _feedforward_values=None,
        dL_dz=None
    ):
        reconstruction_loss = 0
        kl_loss = 0

        mu, log_var, encoder_values = self._encode(
            data_point[0],
            time_separated
        )

        generated, epsilon = self._gen(mu, log_var)

        decoder_loss, dL_dz = self.decoder.backpropagate(
            [generated, data_point[1]],
            print_epochs,
            time_separated_input=False,
            time_separated_output=time_separated,
            num_outputs=len(data_point[0])
        )

        dL_dz = self.gen_backprop(dL_dz, mu, log_var, epsilon)

        reconstruction_loss += decoder_loss[Loss.RECONSTRUCTION_LOSS]
        kl_loss += self.kl_loss(mu, log_var)[Loss.KL_LOSS]

        encoder_loss, _ = self.encoder.backpropagate(
            [data_point[0]],
            # print_epochs is false because there is no objective
            # output for the latent space values.
            False,
            time_separated_input=time_separated,
            time_separated_output=False,
            init_feedforward_values=encoder_values,
            num_outputs=1,
            dL_dz=dL_dz
        )

        return ({
            Loss.RECONSTRUCTION_LOSS: reconstruction_loss,
            Loss.KL_LOSS: kl_loss,
            Loss.TOTAL_LOSS: reconstruction_loss + kl_loss
        }, None)
