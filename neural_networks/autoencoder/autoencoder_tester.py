from neural_networks.autoencoder.autoencoder import AutoEncoder
import numpy as np
from typing import Any

def autoencoder_test(Tester):
  module = Tester("Neural Network AutoEncoder Test")

  network:AutoEncoder = AutoEncoder(layers=(11,1,11))
  input: np.ndarray[Any, np.dtype[np.float64]] = np.random.random((11, 1))
  output = network.feedforward(input)
  module.tester("AutoEncoder Feedforward", module.eq(output.shape, (11, 1)))
