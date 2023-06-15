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
sounds = sounds/10

encoder_layers = (song_length, 50, 25)
decoder_layers = encoder_layers[::-1]
 
losses = []
notANumberCount = 0
for i in range(30):
  compression_VAE = VAE(encoder_layers, decoder_layers)
  compression_VAE.train(sounds, 30, len(sounds), graph=False, learning_rate=0.05, print_epochs=False)

  # print(f"test input:\n {sounds[0]*10}")

  _, _, mu, log_variance = compression_VAE.encode( sounds[0] )

  # print(f"mu:\n {mu} \nsigma:\n {np.exp(0.5*log_variance)}")

  generated, epsilon = compression_VAE.gen(mu, -100)

  # print(f"generated:\n {generated}")

  decoded = compression_VAE.decode(generated)[1]

  # print(f"decoded:\n {decoded[-1]*10}")

  # print(f"off by:\n {song_length*(decoded[-1]-sounds[0])}")
  loss = compression_VAE.vae_loss(sounds[0], decoded[-1], mu, log_variance)
  r_loss = loss[0]

  if math.isnan(r_loss) or r_loss > 1E4:
    notANumberCount+=1
  else:
    losses.append(r_loss)

fig, ax = plt.subplots()
ax: Any = ax
n_bins = 20

# We can set the number of bins with the *bins* keyword argument.
ax.hist(losses, bins=n_bins)
plt.show()
print(f"nan count: {notANumberCount}")
