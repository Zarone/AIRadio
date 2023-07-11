from neural_networks.components.base import BaseNetwork
from neural_networks.components.activations import (
    leaky_relu,
    leaky_relu_derivative,
    linear_derivative
)
from neural_networks.components.optimizer.optimizer import Optimizer
from neural_networks.components.optimizer.adam_w_taperoff import AdamTaperoff
from neural_networks.components.coefs import Coefficients
from neural_networks.components.loss_types import Loss
import numpy as np
from typing import Tuple, List


class Recurrent(BaseNetwork):

    # Just to prevent magic strings constants
    FEEDFORWARD_DATA = {
        "OUTPUT": "OUTPUT",
        "INPUT": "INPUT",
        "HIDDEN_STATE": "HIDDEN_STATE"
    }

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

    def _feedforward(
        self,
        input_val: np.ndarray,
        num_inputs: int,
        num_outputs: int
    ) -> dict:
        """This functions takes an input and returns the z values \
and activations of the network after a feedforward. Returns a tuple \
where the first element is the z layer of the final time step and \
the second element is the a layer of the final time step.

        :param input_val should be a numpy array where the first element is \
the input, and the second element is the true output. Both of these elements \
should be arrays where each element represents a time step.
        """

        # Returned for backpropagation use
        input_processing_data = np.empty((num_outputs))
        hidden_state_processing_data = np.empty((num_outputs))
        output_processing_data = np.empty((num_outputs))

        hidden_state = np.zeros(self.input_layer[-1], 1)

        for i in range(num_outputs):
            time_step = None
            if (num_outputs > num_inputs):
                time_step = input_val[0][0]
            else:
                time_step = input_val[0][i]

            input_value = np.concatenate(hidden_state, time_step)
            input_processing_data[i] = self._custom_feedforward(
                input_value,
                self.input_layers,
                Coefficients.INPUT_WEIGHTS,
                Coefficients.INPUT_BIASES
            )
            input_z_values, input_activations = input_processing_data[i]
            raw_state = input_activations[-1]

            hidden_state_processing_data[i] = self._custom_feedforward(
                raw_state,
                self.state_layers,
                Coefficients.STATE_WEIGHTS,
                Coefficients.STATE_BIASES
            )
            hidden_state_z_values, hidden_state_activations = \
                hidden_state_processing_data[i]
            hidden_state = hidden_state_activations[-1]

            output_processing_data[i] = self._custom_feedforward(
                raw_state,
                self.output_layers,
                Coefficients.OUTPUT_WEIGHTS,
                Coefficients.OUTPUT_BIASES
            )

        return {
            self.FEEDFORWARD_DATA.OUTPUT: output_processing_data,
            self.FEEDFORWARD_DATA.INPUT: input_processing_data,
            self.FEEDFORWARD_DATA.HIDDEN_STATE: hidden_state_processing_data
        }

    def custom_backpropagate(
        self,
        layers,
        start_dL_dz,
        z_values,
        activations,
        weight_name,
        bias_name
    ):
        assert len(z_values) == len(self.coefs[weight_name]) + 1
        assert len(z_values) == len(layers)

        num_layers = len(layers)
        final_layer = num_layers - 1
        dL_dz = start_dL_dz
        for i in range(final_layer, -1, -1):
            self.coef_gradients[weight_name][i-1] += np.matmul(
                dL_dz, activations[i].T
            )
            self.coef_gradients[bias_name][i-1] += dL_dz

            activation_derivative_func = self.activation_derivative\
                if i != final_layer else linear_derivative

            dL_dz = np.matmul(
                self.coefs[weight_name][i-1].T,
                dL_dz
            ) * activation_derivative_func(z_values[i])

        return dL_dz

    def backpropagate(
        self,
        data_point,
        print_epochs,
        dL_dz=None,
        _feedforward_values=None
    ):
        reconstruction_loss = 0

        feedforward_values = _feedforward_values\
            if _feedforward_values is not None\
            else self._feedforward(data_point)

        output_processing_data = feedforward_values[self.FEEDFORWARD_DATA.OUTPUT]
        last_z_values = output_processing_data[-1][0]
        last_activations = output_processing_data[-1][1]
        input_processing_data = feedforward_values[self.FEEDFORWARD_DATA.INPUT]
        hidden_state_processing_data = feedforward_values[self.FEEDFORWARD_DATA.HIDDEN_STATE]

        # Partial Derivative of Loss with respect to the output activations
        dL_daL = dL_dz if dL_dz is not None else\
            (
                (
                    last_activations[-1] - data_point[1][-1]
                ) * (2/len(last_activations[-1]))
            )

        # Backpropagate through output layer, from final output to final state
        dL_dz = self.custom_backpropagate(
            self.output_layers,
            dL_daL,
            last_z_values,
            last_activations,
            Coefficients.OUTPUT_WEIGHTS,
            Coefficients.OUTPUT_BIASES
        )

        num_inputs = len(data_point[0])
        num_outputs = len(data_point[1])

        for i in range(num_inputs-1, -1, -1):
            dL_dz = self.custom_backpropagate(
                self.input_layers,
                dL_dz,
                input_processing_data[i][0],
                input_processing_data[i][1],
                Coefficients.INPUT_WEIGHTS,
                Coefficients.INPUT_BIASES
            )

            dL_dz = dL_dz[0:self.state_layers[-1]]

            dL_dz = self.custom_backpropagate(
                self.state_layers,
                dL_dz,
                hidden_state_processing_data[i][0],
                hidden_state_processing_data[i][1],
                Coefficients.STATE_WEIGHTS,
                Coefficients.STATE_BIASES
            )

            # This indicates that this network is many to many
            # which means that another output must be factored
            # into the gradient calculation.
            if num_outputs > 1:
                true_output = data_point[1][i]
                predicted_output = output_processing_data[i][1][-1]

                dL_dOutput = (
                    predicted_output - true_output
                ) * (2/len(predicted_output))

                dL_dz += self.custom_backpropagate(
                    self.output_layers,
                    dL_dOutput,
                    output_processing_data[i][0],
                    output_processing_data[i][1],
                    Coefficients.OUTPUT_WEIGHTS,
                    Coefficients.OUTPUT_BIASES
                )

        return ({
            Loss.RECONSTRUCTION_LOSS: reconstruction_loss
        }, dL_dz)
