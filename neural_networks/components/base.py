from neural_networks.components.optimizer.optimizer import Optimizer
from neural_networks.components.optimizer.adam_w_taperoff import AdamTaperoff
import neural_networks.components.config as config
from neural_networks.components.activations import (
    leaky_relu_derivative,
    leaky_relu,
    linear,
    linear_derivative
)
from neural_networks.components.loss_types import Loss
from neural_networks.components.coefs import Coefficients
from neural_networks.components.feedforward_data import FeedforwardData
import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Tuple, List
import math


class BaseNetwork:

    """
    :param layers This defines the number of nodes in each \
  activation layer (including input space and latent space)
    :param activation This is the primary activation function \
  which the neural network uses
    """

    def __init__(
        self,
        layers: Tuple[int, ...],
        activation=leaky_relu,
        activation_derivative=leaky_relu_derivative,
        optimizer: Optimizer = AdamTaperoff
    ) -> None:
        self.optimizer = optimizer()
        self.activation = activation
        self.activation_derivative = activation_derivative
        self.layers = layers

        self.coefs = {}
        self.init_coefficients()

        # These are used in training
        self.coef_gradients = {
            Coefficients.WEIGHTS: None,
            Coefficients.BIASES: None
        }

    def get_init_param(self, fan_in, fan_out) -> Tuple[np.ndarray, np.ndarray]:
        """Returns a weight and a bias, generated using the Xavier Glorot \
method which hopes to stabilize network output during both forward and \
backpropagation.

        :param fan_in The number of neurons in the previous layers.
        :param fan_out The number of neurons in this layer.
        """

        standard_deviation = math.sqrt(2/(fan_in + fan_out))

        init_bias = config.rng.normal(
            0, standard_deviation, (fan_out, 1)
        )
        init_weight = config.rng.normal(
            0, standard_deviation, (fan_out, fan_in)
        )

        return (init_weight, init_bias)

    def init_coefficients(self) -> None:
        """This function justs initializes the weights and biases in the \
network according to the layers in the network.
        """
        num_layers = len(self.layers) - 1

        self.coefs[Coefficients.BIASES] = np.array([None] * num_layers)
        self.coefs[Coefficients.WEIGHTS] = np.array([None] * num_layers)

        for i in range(num_layers):
            init_weights, init_bias = self.get_init_param(
                self.layers[i], self.layers[i+1]
            )
            self.coefs[Coefficients.WEIGHTS][i] = init_weights
            self.coefs[Coefficients.BIASES][i] = init_bias

    def _custom_feedforward(
        self,
        input_val: np.ndarray,
        layers,
        weight_name,
        bias_name
    ):
        """This functions takes an input and returns the z values \
and activations of the network after a feedforward. The output does \
not trigger the activation function.

        :param input_val should be a numpy array where the first element is \
the input, and the second element is the true output.
        """

        num_layers = len(layers)
        activations: List = [None] * num_layers
        zs: List = [None] * num_layers

        zs[0] = input_val
        activations[0] = input_val

        for i in range(num_layers-1):
            zs[i+1], activations[i+1] = self.feedforward_custom_layer(
                i,
                activations[i],
                i == num_layers-2,
                weight_name,
                bias_name
            )

        return (zs, activations)

    # I pass in args here just to prevent an error
    # in _feedforward overrides in inherited classes.
    def _feedforward(self, input_val: np.ndarray, **_) -> Tuple[List, List]:
        """This functions takes an input and returns the z values \
and activations of the network after a feedforward.

        :param input_val should be a numpy array where each \
element is an activation value to the first neural network \
layer.
        """

        assert input_val.shape[0] == self.layers[0],\
            f"input_val of unexpected input size {input_val.shape}" + \
            f" when ({self.layers[0]}, 1) was expected"

        output = self._custom_feedforward(
            input_val,
            self.layers,
            Coefficients.WEIGHTS,
            Coefficients.BIASES
        )

        return {
            FeedforwardData.OUTPUT: output
        }

    def feedforward(self, input):
        return self._feedforward(input)[FeedforwardData.OUTPUT][1][-1]

    def feedforward_custom_layer(
        self,
        i: int,
        last_activations: np.ndarray,
        force_linear: bool,
        weight_name: str,
        bias_name: str
    ):
        z = np.matmul(
            self.coefs[weight_name][i], last_activations
        ) + self.coefs[bias_name][i]
        activation_function = self.activation if not force_linear else linear
        return (z, activation_function(z))

    def feedforward_layer(
        self,
        i: int,
        last_activations: np.ndarray,
        force_linear: bool
    ) -> Tuple[np.ndarray, np.ndarray]:
        return self.feedforward_custom_layer(
            i,
            last_activations,
            force_linear,
            Coefficients.WEIGHTS,
            Coefficients.BIASES
        )

    def loss(self, y_true, y_pred):
        n = y_true.shape[0]  # Number of samples

        # Reconstruction loss
        diff = y_true - y_pred
        reconstruction_loss = np.sum(np.square(diff)) / n

        return {
            Loss.RECONSTRUCTION_LOSS: reconstruction_loss
        }

    @staticmethod
    def graph_loss(losses: dict, test_losses={}):
        sub = plt.subplots(2 if not len(test_losses) == 0 else 1, sharex=True)

        axs: Any = sub[1]
        if len(test_losses) == 0:
            axs = [axs]

        for key, value in losses.items():
            axs[0].plot(value, label=key)
        axs[0].set(
            ylabel='Loss',
            title='Loss over time (Training Data)'
        )
        axs[0].legend(loc="upper left")
        axs[0].grid()
        axs[0].semilogy()

        if test_losses:
            for key, value in test_losses.items():
                axs[1].plot(value, label=key)
            axs[1].set(
                xlabel='Mini Batch',
                ylabel='Loss',
                title='Loss over time (Test Data)'
            )
            axs[1].legend(loc="upper left")
            axs[1].grid()
            axs[1].semilogy()

        plt.show()

    @staticmethod
    def split_training_data(data, split=0.1):
        """This method splits the training data into training data \
and test data.

        :param split The percentage of the data that will be commited \
to testing.
        """

        cut_point = math.floor(len(data)*(1-split))
        return data[0:cut_point], data[cut_point:]

    def train(
        self,
        _training_data: np.ndarray,
        max_epochs: int,
        time_seperated_input: bool,
        batch_size: int = 100,
        test_data: (np.ndarray | None) = None,
        learning_rate=0.05,
        graph=False,
        print_epochs=True,
        num_inputs=-1,
        num_outputs=-1
    ) -> None:
        losses = {}
        test_losses = {}

        training_data = np.array(_training_data, copy=True)

        per_epoch = len(training_data) // batch_size

        assert per_epoch != 0, "Batch Size greater than Data Set"

        for i in range(max_epochs):
            np.random.shuffle(training_data)
            for j in range(per_epoch):
                index = j * batch_size
                batch = training_data[index:index+batch_size]

                loss_dictionary = self.training_step(
                    batch,
                    learning_rate,
                    print_epochs,
                    time_seperated_input=time_seperated_input,
                    num_inputs=num_inputs,
                    num_outputs=num_outputs
                )

                for key, value in loss_dictionary.items():
                    losses.setdefault(key, [])
                    losses[key].append(value)

                test_loss = {}
                if not (test_data is None) and print_epochs:
                    for index, element in enumerate(test_data):
                        delta_test_loss = self.loss(
                            test_data[index],
                            self.feedforward(test_data[index])
                        )

                        for key, value in delta_test_loss.items():
                            test_loss.setdefault(key, 0)
                            test_loss[key] += value

                    for key, value in test_loss.items():
                        test_losses.setdefault(key, [])
                        test_losses[key].append(value/len(test_data))

                loss = {key: val[-1] for key, val in losses.items()}
                test_loss = {key: val[-1] for key, val in test_losses.items()}
                if print_epochs:
                    print(
                        f"Epoch {i}, Mini-Batch {j}: " +
                        f"Loss={loss}, " +
                        f"Test Loss={test_loss}"
                    )
        if graph:
            self.graph_loss(losses, test_losses)

    def init_gradients(self):
        for coef in self.coefs.keys():
            if self.coef_gradients[coef] is None:
                self.coef_gradients[coef] = np.empty(
                    self.coefs[coef].shape, np.ndarray
                )
            for i, _ in enumerate(self.coef_gradients[coef]):
                self.coef_gradients[coef][i] = np.zeros(
                    self.coefs[coef][i].shape
                )

    def update_coefficients(self, learning_rate, batch, losses):
        for key, value in self.coefs.items():
            self.coefs[key] -= learning_rate / \
                len(batch) * self.optimizer.adjusted_gradient(
                    key, self.coef_gradients[key],
                    losses[Loss.TOTAL_LOSS]
                    if Loss.TOTAL_LOSS in losses
                    else losses[Loss.RECONSTRUCTION_LOSS]
                )

    def training_step(
        self,
        batch,
        learning_rate,
        print_epochs,
        time_seperated_input,
        num_inputs,
        num_outputs
    ):
        self.init_gradients()

        losses = {}

        for _, data_point in enumerate(batch):
            delta_loss, _ = self.backpropagate(
                data_point,
                print_epochs,
                time_separated=time_seperated_input,
            )
            for key, value in delta_loss.items():
                losses.setdefault(key, 0)
                losses[key] += value

        self.update_coefficients(learning_rate, batch, losses)

        return losses

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
        final_layer = num_layers - 2
        dL_dz = start_dL_dz

        for i in range(final_layer, -1, -1):
            self.coef_gradients[weight_name][i] += np.matmul(
                dL_dz, activations[i].T
            )
            self.coef_gradients[bias_name][i] += dL_dz

            activation_derivative_func = self.activation_derivative\
                if i != final_layer else linear_derivative

            dL_dz = np.matmul(
                self.coefs[weight_name][i].T,
                dL_dz
            ) * activation_derivative_func(z_values[i])

        return dL_dz

    def backpropagate(
        self,
        data_point,
        print_epochs,
        time_separated_input: bool,
        time_separated_output: bool,
        dL_dz=None,
        init_feedforward_values=None,
        num_outputs=1,
    ):
        reconstruction_loss = 0

        feedforward_values = init_feedforward_values[FeedforwardData.OUTPUT]\
            if init_feedforward_values is not None\
            else self._feedforward(data_point[0])[FeedforwardData.OUTPUT]

        z_values = feedforward_values[0]
        activations = feedforward_values[1]

        # Partial Derivative of Loss with respect to the output activations
        dL_daL = dL_dz if dL_dz is not None else\
            (activations[-1] - data_point[1]) * (2/len(activations[-1]))

        if print_epochs:
            delta_reconstruction_loss = self.loss(
                data_point[1], activations[-1]
            )[Loss.RECONSTRUCTION_LOSS]
            reconstruction_loss += delta_reconstruction_loss

        # len_z = len(z_values)

        # decoder_gradients_z = np.array([None] * len_z)

        # decoder_gradients_z[-1] = dL_daL

        # last_index = len_z - 1

        # for j in range(last_index, -1, -1):
            # z_layer = z_values[j]
            # if j != last_index:
                # activation_derivative_func = self.activation_derivative
                # decoder_gradients_z[j] = np.matmul(
                    # self.coefs[Coefficients.WEIGHTS][j+1].transpose(),
                    # decoder_gradients_z[j+1]
                # ) * activation_derivative_func(z_layer)

            # if j != 0:
                # self.coef_gradients[Coefficients.WEIGHTS][j] += np.matmul(
                    # decoder_gradients_z[j], activations[j-1].transpose())
                # self.coef_gradients[Coefficients.BIASES][j] += \
                    # decoder_gradients_z[j]

        # self.coef_gradients[Coefficients.WEIGHTS][0] += np.matmul(
            # decoder_gradients_z[0], data_point[0].transpose()
        # )
        # self.coef_gradients[Coefficients.BIASES][0] += decoder_gradients_z[0]

        # output_dL_dz = np.matmul(
            # self.coefs[Coefficients.WEIGHTS][0].T,
            # decoder_gradients_z[0]
        # )

        output_dL_dz = self.custom_backpropagate(
            self.layers,
            dL_daL,
            z_values,
            activations,
            Coefficients.WEIGHTS,
            Coefficients.BIASES
        )

        return ({
            Loss.RECONSTRUCTION_LOSS: reconstruction_loss
        }, output_dL_dz)

    def save_to_file(self, file: str):
        np.savez(
            file,
            weights=self.coefs[Coefficients.WEIGHTS],
            biases=self.coefs[Coefficients.BIASES]
        )

    def init_from_file(self, file: str):
        parameters = np.load(file, allow_pickle=True)
        self.coefs[Coefficients.WEIGHTS] = parameters['weights']
        self.coefs[Coefficients.BIASES] = parameters['biases']

    @staticmethod
    def format_unsupervised_input(input):
        return np.stack([input, input], axis=1)

    @staticmethod
    def check_vae_compatibility(encoder, decoder):
        assert encoder.layers[-1] == decoder.layers[0] * 2,\
            "Expected final layer in encoder to be twice the" +\
            " first layer in decoder"

        assert encoder.layers[0] == decoder.layers[-1],\
            "Expected the first layer in encoder to be equal to" +\
            " the final layer in the decoder"
