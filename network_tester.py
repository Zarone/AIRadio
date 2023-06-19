import matplotlib.pyplot as plt
from neural_networks.vae.normal_vae.normal_vae import VAE
from neural_networks.components.optimizer.adam import Adam
import math
from typing import Any

def test_network(input, scale, max_epochs, encoder_layers, decoder_layers=None, this_optimizer = Adam, loss_graph = False, histogram: bool = False, histogram_trials = 0, test_data = False):
  decoder_layers = decoder_layers or encoder_layers[::-1]

  sounds = input
  losses = []
  notANumberCount = 0
  for i in range(histogram_trials if histogram else 1):
    if histogram:
      print(f"Training Iteration {i}")
    compression_VAE = VAE(encoder_layers, decoder_layers, optimizer=this_optimizer())
    test_data_start = len(sounds)//4*3 if test_data else len(sounds)
    for i, sound in enumerate(sounds):
      print(f"sound {i}", sound)
    compression_VAE.train(
        sounds[0:test_data_start], 
        max_epochs, 
        test_data_start, 
        graph=(not histogram and loss_graph), 
        learning_rate=0.01, 
        print_epochs=(not histogram ), 
        test_data=sounds[test_data_start:] if test_data else None
      )

    if not histogram:
      print(f"test input:\n {sounds[0]/scale}")

    _, _, mu, log_variance = compression_VAE.encode( sounds[0] )

    generated, epsilon = compression_VAE.gen(mu, log_variance)

    decoded = compression_VAE.decode(generated)[1]

    if not histogram:
      print(f"decoded:\n {decoded[-1]/scale}")

    if not histogram:
      print(f"off by:\n {(decoded[-1]-sounds[0])/scale}")

    loss = compression_VAE.loss(sounds[0]/scale, decoded[-1]/scale, mu, log_variance)
    r_loss = loss[0]

    if math.isnan(r_loss) or r_loss > 1500:
      notANumberCount+=1
    else:
      losses.append(r_loss)

  if histogram:
    fig, ax = plt.subplots()
    ax: Any = ax
    n_bins = 20

    plt.title(f"Training Histogram, Optimizer={this_optimizer.__name__}, nan count: {notANumberCount}")
    ax.hist(losses, bins=n_bins)
    plt.show()
