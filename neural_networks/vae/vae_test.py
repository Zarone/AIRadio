from neural_networks.vae.vae import VAE
from neural_networks.components.base import BaseNetwork
import numpy as np
from typing import Any


def vae_test(Tester):
    with Tester("NN VAE Test") as module:
        network: VAE = VAE(
            encoder_layers=(5, 4, 3),
            decoder_layers=(3, 4, 5),
            sub_network=BaseNetwork
        )
        network.train(
            np.array(
                [1, 2, 3, 4, 5]
            ),
            batch_size=1,
            max_epochs=1,
            graph=True,
            learning_rate=0.01
        )


