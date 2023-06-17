import numpy as np
import audio_parsing.audio_parsing as audio
from neural_networks.components.optimizer.adadelta import Adadelta
from neural_networks.components.optimizer.momentum import Momentum
from neural_networks.components.optimizer.SGD import SGD
from neural_networks.vae.normal_vae.normal_vae import VAE
import math
import matplotlib.pyplot as plt
from typing import Any

sounds, names = audio.get_raw_data()

# This is just a way to test the sound data
# audio.play_random_sound(sounds, names)

song_length = len(sounds[0])
print(f"Current Size of Each Song: {len(sounds[0])}")

# Prevents loss gradients from being too
# extreme
scale = 10
sounds = sounds/scale

encoder_layers = (song_length, 25)
decoder_layers = encoder_layers[::-1]
 
losses = []
notANumberCount = 0
histogram = True

thisOptimizer = Adadelta

for i in range(100 if histogram else 1):
  if histogram:
    print(f"Training Iteration {i}")
  compression_VAE = VAE(encoder_layers, decoder_layers, optimizer=thisOptimizer())
  compression_VAE.train(sounds, 200, len(sounds), graph=(not histogram), learning_rate=0.05, print_epochs=(not histogram))

  if not histogram:
    print(f"test input:\n {sounds[0]*scale}")

  _, _, mu, log_variance = compression_VAE.encode( sounds[0] )

  generated, epsilon = compression_VAE.gen(mu, -100)

  decoded = compression_VAE.decode(generated)[1]

  if not histogram:
    print(f"decoded:\n {decoded[-1]*scale}")

  if not histogram:
    print(f"off by:\n {10*(decoded[-1]-sounds[0])}")

  loss = compression_VAE.loss(sounds[0], decoded[-1], mu, log_variance)
  r_loss = loss[0]

  if math.isnan(r_loss) or r_loss > 15:
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
