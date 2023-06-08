from neural_networks.components.activations import relu_, sigmoid_, relu, sigmoid

def component_test(Tester):
  module = Tester("Neural Network Component Test")
  module.tester("ReLu 1", Tester.eq(relu_(0), 0))
  module.tester("ReLu 2", Tester.eq(relu_(-10), 0))
  module.tester("ReLu 3", Tester.eq(relu_(10), 10))
  module.tester("ReLu 4", Tester.eq(relu([0, -10, 10]), [0, 0, 10]))
  module.tester("Sigmoid 1", Tester.eq(sigmoid_(0), 0.5))
  module.tester("Sigmord 2", Tester.eq(sigmoid_(10), 1))
  module.tester("Sigmord 3", Tester.eq(sigmoid([0, 10]), [0.5, 1]))