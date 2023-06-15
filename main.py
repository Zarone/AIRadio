import numpy as np
import audio_parsing.audio_parsing as audio
from neural_networks.vae.normal_vae.normal_vae import VAE
import math
import matplotlib.pyplot as plt
from typing import Any

sounds, names = audio.get_raw_data()

# This is just a way to test the sound data
# audio.play_random_sound(sounds, names)

# sounds = np.array([
    # np.array([0, 0, 0]).reshape(-1, 1),
    # np.array([0, 0, 1]).reshape(-1, 1),
    # np.array([0, 1, 0]).reshape(-1, 1), 
    # np.array([0, 1, 1]).reshape(-1, 1),  
    # np.array([1, 0, 0]).reshape(-1, 1),  
    # np.array([1, 0, 1]).reshape(-1, 1),  
    # np.array([1, 1, 0]).reshape(-1, 1),  
    # np.array([1, 1, 1]).reshape(-1, 1)
# ])

song_length = len(sounds[0])
print(f"Current Size of Each Song: {len(sounds[0])}")

# Prevents loss gradients from being too
# extreme
scale = 10
sounds = sounds/scale

encoder_layers = (song_length, 80, 60, 50)
decoder_layers = encoder_layers[::-1]
 
losses = []
notANumberCount = 0
histogram = False 

for i in range(30 if histogram else 1):
  compression_VAE = VAE(encoder_layers, decoder_layers)
  compression_VAE.train(sounds, 1000, len(sounds), graph=(not histogram), learning_rate=0.05, print_epochs=(not histogram))

  if not histogram:
    print(f"test input:\n {sounds[0]*scale}")

  _, _, mu, log_variance = compression_VAE.encode( sounds[0] )

  generated, epsilon = compression_VAE.gen(mu, -100)

  decoded = compression_VAE.decode(generated)[1]

  if not histogram:
    print(f"decoded:\n {decoded[-1]*scale}")

  if not histogram:
    print(f"off by:\n {10*(decoded[-1]-sounds[0])}")

  loss = compression_VAE.vae_loss(sounds[0], decoded[-1], mu, log_variance)
  r_loss = loss[0]

  if math.isnan(r_loss) or r_loss > 1E4:
    notANumberCount+=1
  else:
    losses.append(r_loss)

if histogram:
  fig, ax = plt.subplots()
  ax: Any = ax
  n_bins = 20

  ax.hist(losses, bins=n_bins)
  plt.show()
  print(f"nan count: {notANumberCount}")
