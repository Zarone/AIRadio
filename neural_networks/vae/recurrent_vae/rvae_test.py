from neural_networks.components.activations import *
from neural_networks.vae.recurrent_vae.recurrent_vae import RecurrentVAE 
import numpy as np

def rvae_test(Tester):
  module = Tester("Recurrent VAE Test")

  sounds = np.random.random((3, 10, 1))
  network: RecurrentVAE = RecurrentVAE((5, 4, 3, 3), (3, 3, 4, 5))
  time_seperated_sounds: np.ndarray = network.get_time_seperated_data(sounds)
  encoded = network.encode(time_seperated_sounds[0])
  _, _, mu, logvar, _  = encoded
  generated, _ = network.gen(mu, logvar)
  decoded = network.decode(generated, 2)

  module.tester("RVAE Decoded Test 1", module.eq(decoded.shape, (2, ) ))
  module.tester("RVAE Decoded Test 2", module.eq(decoded[0].shape, (5, 1) ))
