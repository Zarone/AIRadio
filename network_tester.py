import matplotlib.pyplot as plt
from neural_networks.vae.normal_vae.normal_vae import VAE
from neural_networks.components.optimizer.adam import Adam
import math
from typing import Any

def test_network(input, scale, max_epochs, encoder_layers, decoder_layers=None, thisOptimizer = Adam, histogram: bool = False, histogram_trials = 0):
  decoder_layers = decoder_layers or encoder_layers[::-1]

  sounds = input * scale
  losses = []
  notANumberCount = 0
  for i in range(histogram_trials if histogram else 1):
    if histogram:
      print(f"Training Iteration {i}")
    compression_VAE = VAE(encoder_layers, decoder_layers, optimizer=thisOptimizer())
    compression_VAE.train(sounds, max_epochs, len(sounds), graph=(not histogram), learning_rate=0.1, print_epochs=(not histogram))

    if not histogram:
      print(f"test input:\n {sounds[0]*scale}")

    _, _, mu, log_variance = compression_VAE.encode( sounds[0] )

    generated, epsilon = compression_VAE.gen(mu, log_variance)

    decoded = compression_VAE.decode(generated)[1]

    if not histogram:
      print(f"decoded:\n {decoded[-1]/scale}")

    if not histogram:
      print(f"off by:\n {(decoded[-1]-sounds[0])/scale}")

    loss = compression_VAE.loss(sounds[0]*10, decoded[-1]*10, mu, log_variance)
    r_loss = loss[0]

    if math.isnan(r_loss) or r_loss > 1500:
      notANumberCount+=1
    else:
      losses.append(r_loss)

  if histogram:
    fig, ax = plt.subplots()
    ax: Any = ax
    n_bins = 20

    plt.title(f"Training Histogram, Optimizer={thisOptimizer.__name__}, nan count: {notANumberCount}")
    ax.hist(losses, bins=n_bins)
    plt.show()
