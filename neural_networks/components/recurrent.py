from neural_networks.components.base import BaseNetwork
from neural_networks.components.activations import (
    leaky_relu, leaky_relu_derivative
)
from neural_networks.components.optimizer.optimizer import Optimizer
from neural_networks.components.optimizer.adam_w_taperoff import AdamTaperoff
from neural_networks.components.coefs import Coefficients
import numpy as np
from typing import Tuple, List


class Recurrent(BaseNetwork):
    def __init__(
        self,
        layers: Tuple[int, ...],
        input_layers: Tuple[int, ...] = None,
        output_layers: Tuple[int, ...] = None,
        state_layers: Tuple[int, ...] = None,
        activation=leaky_relu,
        activation_derivative=leaky_relu_derivative,
        optimizer: Optimizer = AdamTaperoff
    ) -> None:
        self.optimizer = optimizer()
        self.activation = activation
        self.activation_derivative = activation_derivative

        if input_layers is None:
            self.input_layers = input_layers[:-1]
        else:
            self.input_layers = input_layers

        if output_layers is None:
            self.output_layers = (input_layers[-2], input_layers[-1])
        else:
            self.output_layers = output_layers

        if state_layers is None:
            self.state_layers = (input_layers[-2], input_layers[-2])
        else:
            self.state_layers = state_layers

        self.coefs = {}
        self.init_coefficients()

        # These are used in training
        self.coef_gradients = {}

    def init_coefficients_from_layer(
        self,
        layers: np.ndarray,
        weight_name,
        bias_name
    ):
        num_layers = len(layers) - 1

        self.coefs[weight_name] = np.array(
            [None] * num_layers
        )
        self.coefs[bias_name] = np.array(
            [None] * num_layers
        )

        for i in range(num_layers):
            init_weights, init_bias = self.get_init_param(
                layers[i], layers[i+1]
            )
            self.coefs[weight_name][i] = init_weights
            self.coefs[bias_name][i] = init_bias

    def init_coefficients(self) -> None:
        """This function justs initializes the weights and biases in the \
network according to the layers in the network.
        """
        self.init_coefficients_from_layer(
            self.input_layers,
            Coefficients.INPUT_WEIGHTS,
            Coefficients.INPUT_BIASES
        )
        self.init_coefficients_from_layer(
            self.output_layers,
            Coefficients.OUTPUT_WEIGHTS,
            Coefficients.OUTPUT_BIASES
        )
        self.init_coefficients_from_layer(
            self.state_layers,
            Coefficients.STATE_WEIGHTS,
            Coefficients.STATE_BIASES
        )

    def _feedforward(self, input_val: np.ndarray) -> Tuple[List, List]:
        """This functions takes an input and returns the z values \
and activations of the network after a feedforward. Returns a tuple \
where the first element is the z layer of the final time step and \
the second element is the a layer of the final time step.

        :param input_val should be a numpy array where the first element is \
the input, and the second element is the true output.
        """

        """
        hidden_state = np.zeros(self.input_layer[-1], 1)

        for each time step:
            next_input = np.concatenate(input_val[0][time step], hidden_state)

            raw_state self._custom_feedforward(self.input_layers, next_input)
            hidden_state = self._custom_feedforward(self.state_layers, raw_state)
            output = self._custom_feedforward(self.output_layers)

        """
