import numpy as np
import audio_parsing.audio_parsing as audio
from neural_networks.vae.normal_vae.normal_vae import VAE

# sounds, names = audio.get_raw_data()

# This is just a way to test the sound data
# audio.play_random_sound(sounds, names)

# print(f"Current Size of Each Song: {len(sounds[0])}")

layers = (5, 3, 1, 3, 5)
compression_VAE = VAE(layers)
compression_VAE.train()
encoded = compression_VAE.encode(np.arange(len(layers)).reshape(-1,1))
print(f"encoded: {encoded}")
decoded = compression_VAE.decode(encoded)
print(f"decoded: {decoded}")
# compressed_sounds = np.asarray([compression_VAE.encode(sound) for sound in sounds])

