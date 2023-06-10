from neural_networks.components.activations import sigmoid_, relu, sigmoid
import numpy as np

def component_test(Tester):
  module = Tester("Neural Network Component Test")
  module.tester("ReLu 1", Tester.eq(relu([0]), [0]))
  module.tester("ReLu 2", Tester.eq(relu([-10]), [0]))
  module.tester("ReLu 3", Tester.eq(relu([10]), [10]))
  module.tester("ReLu 4", Tester.eq(relu([0, -10, 10]), [0, 0, 10]))
  module.tester("Sigmoid 1", Tester.eq(sigmoid_(0), 0.5))
  module.tester("Sigmoid 2", Tester.eq(sigmoid_(10), 1))
  module.tester("Sigmoid 3", Tester.eq(sigmoid([0, 10]), [0.5, 1]))
