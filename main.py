import numpy as np
import audio_parsing.audio_parsing as audio
from neural_networks.vae.normal_vae.normal_vae import VAE

# sounds, names = audio.get_raw_data()

# This is just a way to test the sound data
# audio.play_random_sound(sounds, names)

# print(f"Current Size of Each Song: {len(sounds[0])}")

# TODO
compression_VAE = VAE([5, 3, 2, 3, 5])
compression_VAE.train()
print(compression_VAE.feedforward(np.arange(5).reshape(-1,1)))
# compressed_sounds = np.asarray([compression_VAE.encode(sound) for sound in sounds])

