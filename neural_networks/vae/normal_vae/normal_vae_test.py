from neural_networks.components.activations import *
from neural_networks.vae.normal_vae.normal_vae import VAE
import numpy as np
from typing import Any 

def normal_vae_test(Tester):
  module = Tester("NN VAE Test")

  network:VAE = VAE(encoder_layers=(10,1), decoder_layers=(1,10))
  input: np.ndarray[Any, np.dtype[np.float64]] = np.random.random((10, 1))
  output = network.feedforward(input)
  module.tester("VAE Feedforward", module.eq(output.shape, (10, 1)))
