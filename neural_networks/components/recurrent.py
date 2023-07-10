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

    def get_time_seperated_data(self, input_data):
        """This function divides the input data into evenly sized vectors\
    with one for each time step.
        """
        input_layer_size = self.encoder_layers[0]
        input_data_size = input_data[0].shape[0]

        assert input_data_size % input_layer_size == 0,\
            "Input data cannot be divided evenly into input layer"

        return_array = np.empty((len(input_data),), dtype=np.ndarray)
        for i, data_point in enumerate(input_data):
            return_array[i] = data_point.reshape(
                data_point.shape[0]//input_layer_size, input_layer_size, 1)

        return return_array

    def _feedforward(self, input_val: np.ndarray) -> Tuple[List, List]:
        """This functions takes an input and returns the z values \
and activations of the network after a feedforward. Returns a tuple \
where the first element is the z layer of the final time step and \
the second element is the a layer of the final time step.

        :param input_val should be a numpy array where the first element is \
the input, and the second element is the true output.
        """

        hidden_state = np.zeros(self.input_layer[-1], 1)
        output = np.empty((len(input_val[0])))
        for i, time_step in enumerate(input_val[0]):
            next_input = np.concatenate(time_step, hidden_state)

            raw_state = self._custom_feedforward(
                next_input,
                self.input_layers,
                Coefficients.INPUT_WEIGHTS,
                Coefficients.INPUT_BIASES
            )
            hidden_state = self._custom_feedforward(
                raw_state,
                self.state_layers,
                Coefficients.STATE_WEIGHTS,
                Coefficients.STATE_BIASES
            )
            feedforward_output = self._custom_feedforward(
                raw_state,
                self.output_layers,
                Coefficients.OUTPUT_WEIGHTS,
                Coefficients.OUTPUT_BIASES
            )
            output[i] = feedforward_output

        return output[-1][0], output[-1][1]
