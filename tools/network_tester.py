from neural_networks.components.base import BaseNetwork
import matplotlib.pyplot as plt
from neural_networks.vae.normal_vae.normal_vae import VAE
from neural_networks.components.optimizer.adam import Adam
import math
from typing import Any, Dict

def test_network(network_class, args, input, scale, max_epochs, this_optimizer = Adam, loss_graph = False, histogram: bool = False, histogram_trials = 0, test_data = False):
  sounds = input
  losses = []
  notANumberCount = 0

  activation_name = ""

  for i in range(histogram_trials if histogram else 1):
    if histogram:
      print(f"Training Iteration {i}")
    network = network_class( **args )

    # Dedicates 1/4th of the data set to test data if test_data is True
    test_data_start = len(sounds)//4*3 if test_data else len(sounds)

    network.train(
        sounds[0:test_data_start], 
        max_epochs, 
        test_data_start, 
        graph=(not histogram and loss_graph), 
        learning_rate=0.001, 
        print_epochs=(not histogram ), 
        test_data=sounds[test_data_start:] if test_data else None
      )
    activation_name = network.activation.__name__

    if not histogram:
      print(f"test input:\n {sounds[0]/scale}")

    decoded = network.feedforward(sounds[0])

    if not histogram:
      print(f"decoded:\n {decoded/scale}")

    if not histogram:
      print(f"off by:\n {(decoded-sounds[0])/scale}")

    loss = network.loss(sounds[0]/scale, decoded/scale)
    r_loss = loss[0]
    print("reconstruction loss", r_loss)

    # totally arbitrary number
    threshold = 5E4

    if math.isnan(r_loss) or r_loss > threshold:
      notANumberCount+=1
    else:
      losses.append(r_loss)

  if histogram:
    fig, ax = plt.subplots()
    ax: Any = ax
    n_bins = 20

    plt.title(f"Training Histogram, Optimizer={this_optimizer.__name__}, Activation={activation_name}, nan count: {notANumberCount}")
    ax.hist(losses, bins=n_bins)
    plt.show()

def test_vae(input, scale, max_epochs, layers, this_optimizer = Adam, loss_graph = False, histogram: bool = False, histogram_trials = 0, test_data = False):
  encoder_layers = layers[:math.ceil(len(layers)/2)]
  decoder_layers = layers[math.floor(len(layers)/2):]
  args: Dict = {
      "encoder_layers": encoder_layers, 
      "decoder_layers": decoder_layers, 
      "optimizer": this_optimizer(),
    }
  test_network(VAE, args, input, scale, max_epochs, this_optimizer, loss_graph, histogram, histogram_trials, test_data)

def test_base(input, scale, max_epochs, layers, this_optimizer = Adam, loss_graph = False, histogram: bool = False, histogram_trials = 0, test_data = False):
    args: Dict = {
        "layers": layers, 
        "optimizer": this_optimizer(),
      }
    test_network(BaseNetwork, args, input, scale, max_epochs, this_optimizer, loss_graph, histogram, histogram_trials, test_data)

