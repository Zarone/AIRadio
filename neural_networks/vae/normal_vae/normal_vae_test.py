from neural_networks.components.activations import *
from neural_networks.vae.normal_vae.normal_vae import VAE
import numpy as np
from typing import Any


def normal_vae_test(Tester):
    with Tester("NN VAE Test") as module:
        network: VAE = VAE(encoder_layers=(3, 2, 1), decoder_layers=(1, 2, 3))
        input: np.ndarray[Any, np.dtype[np.float64]] = np.random.random((3, 1))
        output = network.feedforward(input)

        module.tester("VAE Feedforward", module.eq(output.shape, (3, 1)))

        network.train(
            np.array(
                [
                    [[1], [2], [3]],
                    [[4], [5], [6]],
                    [[7], [8], [9]]
                ]
            ), 1, 3, print_epochs=False
        )
