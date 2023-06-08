import numpy as np
from neural_networks.vae.normal_vae.normal_vae import VAE

def nn_VAE_test(Tester):
  module = Tester("Neural Network VAE Test")

  layers = (5,3,1,3,5)
  my_VAE = VAE(layers)
  input = np.arange(len(layers))
  module.tester(
    "Encoder Decoder 1", 
    Tester.eq(
      my_VAE.decode(my_VAE.encode(input)), 
      my_VAE.feedforward(input)
    )
  )

  layers = (3,1,3)
  my_VAE = VAE(layers)
  input = np.arange(len(layers))
  module.tester(
    "Encoder Decoder 2", 
    Tester.eq(
      my_VAE.decode(my_VAE.encode(input)), 
      my_VAE.feedforward(input)
    )
  )
