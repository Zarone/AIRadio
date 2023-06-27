from neural_networks.autoencoder.autoencoder import AutoEncoder
import numpy as np
from typing import Any

def autoencoder_test(Tester):
  module = Tester("Neural Network AutoEncoder Test")

  network:AutoEncoder = AutoEncoder(layers=(3,1,3))
  input: np.ndarray[Any, np.dtype[np.float64]] = np.random.random((3, 1))
  output = network.feedforward(input)
  module.tester("AutoEncoder Feedforward", module.eq(output.shape, (3, 1)))

  network.train( np.array( [ [[1], [2], [3]], [[4], [5], [6]], [[7], [8], [9]] ] ), 1, 3, print_epochs=False )
