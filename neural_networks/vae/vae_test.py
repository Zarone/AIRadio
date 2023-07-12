from neural_networks.vae.vae import VAE
from neural_networks.components.base import BaseNetwork
from neural_networks.components.recurrent import Recurrent
import numpy as np
from typing import Any


def vae_test(Tester):
    with Tester("NN VAE Test") as module:
        test_sounds = np.array(
            [
                [
                    [1], [2], [3], [4], [5], [6], [7], [8], [9], [10]
                ]
            ]
        )
        network: VAE = VAE(
            encoder_layers=(10, 4, 3),
            decoder_layers=(3, 4, 10),
            sub_network=BaseNetwork
        )
        network.train(
            test_sounds,
            batch_size=1,
            max_epochs=1,
            graph=False,
            print_epochs=False,
            learning_rate=0.01
        )

        network: VAE = VAE(
            encoder_layers=(5, 4, 3),
            decoder_layers=(3, 4, 5),
            sub_network=Recurrent
        )
        network.train(
            network.get_time_seperated_data(test_sounds),
            batch_size=1,
            max_epochs=1,
            graph=False,
            print_epochs=False,
            learning_rate=0.01
        )
