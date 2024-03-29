from neural_networks.components.activations import (
    relu,
    relu_derivative,
    leaky_relu,
    leaky_relu_derivative,
    elu,
    elu_derivative,
    sigmoid,
    sigmoid_derivative,
    linear,
    linear_derivative
)
from neural_networks.components.base import BaseNetwork
from neural_networks.components.loss_types import Loss
from neural_networks.components.coefs import Coefficients
from neural_networks.components.feedforward_data import FeedforwardData
import numpy as np


def component_test(Tester):
    with Tester("Neural Network Component Test") as module:
        module.tester("ReLu 1", module.eq(relu(np.array([0])), [0]))
        module.tester("ReLu 2", module.eq(relu(np.array([-10])), [0]))
        module.tester("ReLu 3", module.eq(relu(np.array([10])), [10]))
        module.tester("ReLu 4", module.eq(
            relu(np.array([0, -10, 10])), [0, 0, 10]))

        module.tester("ReLu Derivative 1", module.eq(
            relu_derivative(np.array([0, 10])), [0, 1]))
        module.tester("ReLu Derivative 2", module.eq(
            relu_derivative(np.array([-5, 5])), [0, 1]))
        module.tester("ReLu Derivative 3", module.eq(
            relu_derivative(np.array([-10, -1])), [0, 0]))

        module.tester("Leaky ReLu 1", module.eq(
            leaky_relu(np.array([0, 10])), [0, 10]))
        module.tester("Leaky ReLu 2", module.eq(
            leaky_relu(np.array([-5, 5])), [-0.05, 5]))
        module.tester("Leaky ReLu 3", module.eq(
            leaky_relu(np.array([-10, -1])), [-0.1, -0.01]))

        module.tester("Leaky ReLu Derivative 1", module.eq(
            leaky_relu_derivative(np.array([0, 10])), [1, 1]))
        module.tester("Leaky ReLu Derivative 2", module.eq(
            leaky_relu_derivative(np.array([-5, 5])), [0.01, 1]))
        module.tester("Leaky ReLu Derivative 3", module.eq(
            leaky_relu_derivative(np.array([-10, -1])), [0.01, 0.01]))

        module.tester("ELU 1", module.eq(elu(np.array([0, 10])), [0, 10]))
        module.tester("ELU 2", module.eq(elu(np.array([-5, 5])), [-0.99326205, 5]))
        module.tester("ELU 3", module.eq(
            elu(np.array([-10, -1])), [-0.99995460, -0.63212056]))

        module.tester("ELU Derivative 1", module.eq(
            elu_derivative(np.array([0, 10])), [1, 1]))
        module.tester("ELU Derivative 2", module.eq(
            elu_derivative(np.array([-5, 5])), [0.00673795, 1]))
        module.tester("ELU Derivative 3", module.eq(elu_derivative(
            np.array([-10, -1])), [4.53999298e-05, 0.36787944]))

        module.tester("Sigmoid 1", module.eq(sigmoid(0), 0.5))
        module.tester("Sigmoid 2", module.eq(sigmoid(10), 1))
        module.tester("Sigmoid 3", module.eq(sigmoid(np.array([0, 10])), [0.5, 1]))

        module.tester("Sigmoid Derivative 1", module.eq(
            sigmoid_derivative(np.array([0, 1])), [0.25, 0.19661193]))
        module.tester("Sigmoid Derivative 2", module.eq(
            sigmoid_derivative(np.array([-1, 2])), [0.19661193, 0.10499359]))
        module.tester("Sigmoid Derivative 3", module.eq(
            sigmoid_derivative(np.array([-2, -3])), [0.10499359, 0.04517666]))

        module.tester("Linear 1", module.eq(linear(np.array([0, 10])), [0, 10]))
        module.tester("Linear Derivative 1", module.eq(
            linear_derivative(np.array([0, 10])), [1, 1]))

        # Test for linear layer initialization
        network = BaseNetwork(layers=(2, 3, 1))
        module.tester("Parameter Shape 1", module.eq(network.coefs[Coefficients.BIASES].shape, (2,)))
        module.tester("Parameter Shape 2", module.eq(network.coefs[Coefficients.WEIGHTS].shape, (2,)))
        module.tester("Parameter Shape 3", module.eq(
            network.coefs[Coefficients.BIASES][0].shape, (3, 1)))
        module.tester("Parameter Shape 4", module.eq(
            network.coefs[Coefficients.WEIGHTS][0].shape, (3, 2)))
        module.tester("Parameter Shape 5", module.eq(
            network.coefs[Coefficients.BIASES][1].shape, (1, 1)))
        module.tester("Parameter Shape 6", module.eq(
            network.coefs[Coefficients.WEIGHTS][1].shape, (1, 3)))

        # Test for feedforward layer
        network.coefs[Coefficients.WEIGHTS] = np.array((1,), dtype=np.ndarray)
        network.coefs[Coefficients.BIASES] = np.array((1,), dtype=np.ndarray)
        network.coefs[Coefficients.WEIGHTS][0] = np.array([[1, 2], [3, 4]])
        network.coefs[Coefficients.BIASES][0] = np.array([1, -1])
        inputs = np.array([3, 2])
        expected_z = np.array([8, 16])
        expected_activation = np.array([8, 16])
        module.tester("Network Feedforward 1", module.eq(
                network.feedforward_layer(0, inputs, False),
                (expected_z, expected_activation)
            )
        )

        # Test for feedforward full
        network.layers = (2, 2)
        inputs = np.array([3, 2])
        expected_zs = [np.array([3, 2]), np.array([8, 16])]
        expected_activations = [np.array([3, 2]), np.array([8, 16])]
        true_output = network._feedforward(inputs)
        module.tester(
            "Network Feedforward 2",
            module.eq(
                true_output[FeedforwardData.OUTPUT],
                (expected_zs, expected_activations)
            )
        )

        # Test for feedforward
        inputs = np.array([3, 2])
        expected_output = np.array([8, 16])
        module.tester(
            "Network Feedforward 3",
            module.eq(
                network.feedforward(inputs),
                expected_output
            )
        )

        # Test for loss function
        y_true = np.array([1, 0])
        y_pred = np.array([0.2, 0.7])
        expected_loss = np.sum(np.square(y_true - y_pred)) / len(y_true)
        module.tester(
            "Network Loss",
            module.eq(
                network.loss(y_true, y_pred),
                {Loss.RECONSTRUCTION_LOSS: expected_loss}
            )
        )

        network.train(
            np.array(
                [
                    [[1, 2], [2, 4]],
                    [[3, 4], [6, 8]],
                    [[5, 6], [10, 12]]
                ]
            ),
            max_epochs=1,
            batch_size=3,
            print_epochs=False,
            time_separated_input=False
        )
