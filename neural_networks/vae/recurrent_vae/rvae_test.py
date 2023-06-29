from neural_networks.components.activations import *
from neural_networks.vae.recurrent_vae.recurrent_vae import RecurrentVAE
import numpy as np


def rvae_test(Tester):
    with Tester("Recurrent VAE Test") as module:
        sounds = np.random.random((3, 10, 1))
        network: RecurrentVAE = RecurrentVAE((5, 4, 3, 3), (3, 3, 4, 5))
        time_seperated_sounds: np.ndarray = network\
            .get_time_seperated_data(sounds)
        encoded = network.encode(time_seperated_sounds[0])
        mu, logvar = encoded

        module.tester("RVAE Encoder Test 1", module.eq(mu.shape, (3, 1)))
        module.tester("RVAE Encoder Test 2", module.eq(logvar.shape, (3, 1)))

        generated = network.gen(mu, logvar)
        module.tester(
            "RVAE Generator Test 1",
            module.eq(generated.shape, (3, 1))
        )

        decoded = network.decode(generated, 2)

        module.tester(
            "RVAE Decoder Test 1",
            module.eq(decoded.shape, (2, ))
        )
        module.tester(
            "RVAE Decoder Test 2",
            module.eq(decoded[0].shape, (5, 1))
        )
