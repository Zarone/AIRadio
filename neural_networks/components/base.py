from neural_networks.components.optimizer.optimizer import Optimizer
from neural_networks.components.optimizer.adam import Adam
from typing import Any, Tuple, List
import numpy as np
import neural_networks.components.config as config
from neural_networks.components.activations import\
    leaky_relu_derivative, leaky_relu, linear
import math
import matplotlib.pyplot as plt


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
        optimizer: Optimizer = Adam(loss_taperoff=True)
    ) -> None:
        self.optimizer = optimizer
        self.activation = activation
        self.activation_derivative = activation_derivative

        self.init_coefficients(layers)

        # These are used in training
        self.weight_gradient = None
        self.bias_gradient = None

    def get_init_param(self, layer_size):
        max = math.sqrt(2/layer_size)
        return max

    def init_coefficients(self, layers: Tuple[int, ...]) -> None:
        self.layers = layers
        num_layers = len(self.layers) - 1

        self.biases: np.ndarray = np.array([None] * num_layers)
        self.weights: np.ndarray = np.array([None] * num_layers)

        for i in range(num_layers):
            max = self.get_init_param(layers[i])
            self.biases[i] = config.rng.normal(
                0, max, (layers[i+1], 1)
            )
            self.weights[i] = config.rng.normal(
                0, max, (layers[i+1], layers[i])
            )

    def _feedforward(self, input_val: np.ndarray) -> Tuple[List, List]:
        """This functions takes an input and returns the z values 
        and activations of the network after a feedforward.

        :param input_val should be a numpy array where the first element is the input, \
    and the second element is the true output.
        """

        assert len(
            input_val.shape) != 1, f"function expected input_val shape of (n,2), received shape of {input_val.shape}"

        num_layers = len(self.layers) - 1
        activations: List = [None] * num_layers
        zs: List = [None] * num_layers

        last_activations = input_val[0]
        for i, _ in enumerate(activations):
            zs[i], activations[i] = self.feedforward_layer(i, last_activations, i==len(activations)-1)
            last_activations = activations[i]

        return (zs, activations)

    def feedforward(self, input):
        return self._feedforward(input)[1][-1]

    def feedforward_layer(self, i: int, last_activations: np.ndarray, force_linear: bool) -> Tuple[np.ndarray, np.ndarray]:
        # z_{i} = w * a_{i-1} + b
        z = np.matmul(self.weights[i], last_activations) + self.biases[i]

        activation_function = self.activation if not force_linear else linear
        return (z, activation_function(z))

    def loss(self, y_true, y_pred):
        assert len(
            y_true.shape) != 1, f"function expected input_val shape of (n,2), received shape of {y_true.shape}"

        n = y_true[1].shape[0]  # Number of samples

        diff = y_true[1] - y_pred

        # Reconstruction loss
        reconstruction_loss = np.sum(np.square(diff)) / n

        return (reconstruction_loss,)

    @staticmethod
    def graph_loss(losses, test_losses=[]):
        sub = plt.subplots(2 if not len(test_losses) == 0 else 1, sharex=True)
        axs: Any = sub[1]
        if len(test_losses) == 0:
            axs = [axs]
        axs[0].plot(losses, "purple", label="Total Loss")
        axs[0].set(ylabel='Loss',
                   title='Loss over time (Training Data)')
        axs[0].legend(loc="upper left")
        axs[0].grid()
        axs[0].semilogy()

        if test_losses:
            axs[1].plot(test_losses, "purple", label="Total Loss")
            axs[1].set(xlabel='Mini Batch', ylabel='Loss',
                       title='Loss over time (Test Data)')
            axs[1].legend(loc="upper left")
            axs[1].grid()
            axs[1].semilogy()

        plt.show()

    def train(
        self, 
        _training_data: np.ndarray, 
        max_epochs: int, 
        batch_size: int = 100, 
        test_data: (np.ndarray | None) = None, 
        learning_rate=0.05, 
        graph=False, 
        print_epochs=True
    ) -> None:
        losses = []
        test_losses = []

        training_data = np.array(_training_data, copy=True)

        assert training_data.shape[1] == 2\
            and training_data.shape[2] == self.layers[0],\
            f"Expected shape of (N, 2, {self.layers[0]}, 1), but got {training_data.shape}"

        per_epoch = len(training_data) // batch_size

        assert per_epoch != 0, "Batch Size greater than Data Set"
        for i in range(max_epochs):
            np.random.shuffle(training_data)
            for j in range(per_epoch):
                index = j * batch_size
                batch = training_data[index:index+batch_size]

                reconstruction_loss = self.training_step(
                    batch, learning_rate, print_epochs)[0]
                loss = reconstruction_loss
                losses.append(loss)

                test_loss = 0
                if not (test_data is None) and print_epochs:
                    for index, element in enumerate(test_data):
                        delta_test_reconstruction_loss = self.loss(
                            test_data[index], self.feedforward(test_data[index]))[0]
                        test_loss += delta_test_reconstruction_loss

                    test_loss /= len(test_data)
                    test_losses.append(test_loss)

                if print_epochs:
                    print(
                        f"Epoch {i}, Mini-Batch {j}: Loss = {loss}, Test Loss = {test_loss}")
        if graph:
            self.graph_loss(losses, test_losses)

    def init_gradients(self):
        if self.weight_gradient is None or self.bias_gradient is None:
            self.weight_gradient = np.empty(self.weights.shape, np.ndarray)
            self.bias_gradient = np.empty(self.biases.shape, np.ndarray)
        for i, _ in enumerate(self.weight_gradient):
            self.weight_gradient[i] = np.zeros(self.weights[i].shape)
            self.bias_gradient[i] = np.zeros(self.biases[i].shape)

    def training_step(self, batch, learning_rate, print_epochs):
        self.init_gradients()

        assert (not self.weight_gradient is None and not self.bias_gradient is None), "Weight gradient not defined for some reason"

        reconstruction_loss = 0

        len_layers = len(self.layers)
        for _, data_point in enumerate(batch):
            z_values, activations = self._feedforward(data_point)

            # Partial Derivative of Loss with respect to the output activations
            dL_daL = (activations[-1] - data_point[1]) * \
                (2/len(activations[-1]))
            if print_epochs:
                delta_reconstruction_loss = self.loss(
                    data_point, activations[-1])[0]
                reconstruction_loss += delta_reconstruction_loss

            len_z = len(z_values)

            decoder_gradients_z = np.array([None] * len_z)

            decoder_gradients_z[-1] = dL_daL  # * linear_derivative(z_values[-1])

            last_index = len_z - 1

            for j in range(last_index, -1, -1):
                z_layer = z_values[j]
                if j != last_index:
                    activation_derivative_func = self.activation_derivative
                    decoder_gradients_z[j] = np.matmul(
                        self.weights[j+1].transpose(), decoder_gradients_z[j+1]
                    ) * activation_derivative_func(z_layer)

                if j != 0:
                    self.weight_gradient[j] += np.matmul(
                        decoder_gradients_z[j], activations[j-1].transpose())
                    self.bias_gradient[j] += decoder_gradients_z[j]

            self.weight_gradient[0] += np.matmul(
                decoder_gradients_z[0], data_point[0].transpose())
            self.bias_gradient[0] += decoder_gradients_z[0]

        self.weights -= learning_rate / \
            len(batch) * self.optimizer.adjusted_weight_gradient(self.weight_gradient, reconstruction_loss)
        self.biases -= learning_rate / \
            len(batch) * self.optimizer.adjusted_bias_gradient(self.bias_gradient, reconstruction_loss)
        return (reconstruction_loss/len(batch),)

    @staticmethod
    def format_unsupervised_input(input):
        return np.stack([input, input], axis=1)

    def save_to_file(self, file: str):
        np.savez(file, weights=self.weights, biases=self.biases)

    def init_from_file(self, file: str):
        parameters = np.load(file, allow_pickle=True)
        self.weights = parameters['weights']
        self.biases = parameters['biases']
