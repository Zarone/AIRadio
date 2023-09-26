from neural_networks.vae.vae import VAE
from neural_networks.components.base import BaseNetwork
from neural_networks.components.recurrent import Recurrent
import numpy as np
from typing import Any


def vae_test(Tester):
    with Tester("NN VAE Test") as module:
        test_sounds_5 = np.array(
            [
                [
                    [1], [2], [3], [4], [5]
                ]
            ]
        )

        test_sounds_10 = np.array(
            [
                [
                    [1], [2], [3], [4], [5], [6], [7], [8], [9], [10]
                ]
            ]
        )

        network: VAE = VAE(
            encoder_args=dict(layers=(5, 4, 6)),
            decoder_args=dict(layers=(3, 4, 5)),
            sub_network=BaseNetwork,
            latent_size=3
        )

        network.train(
            test_sounds_5,
            batch_size=1,
            max_epochs=1,
            graph=False,
            print_epochs=False,
            learning_rate=0.01,
            time_separated_input=False,
            time_separated_output=False
        )

        network: VAE = VAE(
            encoder_args=dict(
                input_size=5,
                input_layers=(9, 4),
                output_layers=(4, 6)
            ),
            decoder_args=dict(
                input_size=3,
                input_layers=(7, 4),
                output_layers=(4, 5)
            ),
            latent_size=3,
            sub_network=Recurrent
        )

        network.train(
            network.get_time_seperated_data(test_sounds_10),
            batch_size=1,
            max_epochs=1,
            graph=False,
            print_epochs=False,
            learning_rate=0.01,
            time_separated_input=True,
            time_separated_output=True
        )
