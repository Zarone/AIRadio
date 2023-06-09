import numpy as np
import audio_parsing.audio_parsing as audio
from neural_networks.vae.normal_vae.normal_vae import VAE

# sounds, names = audio.get_raw_data()

# This is just a way to test the sound data
# audio.play_random_sound(sounds, names)

# print(f"Current Size of Each Song: {len(sounds[0])}")

encoder_layers = (5, 3, 1)
decoder_layers = (1, 3, 5)

compression_VAE = VAE(encoder_layers, decoder_layers)
_, _, mu, log_variance = compression_VAE.encode( np.arange(5).reshape(-1,1) )
print(f"mu:\n {mu}, \nsigma:\n {np.exp(0.5*log_variance)}")
generated = compression_VAE.gen(mu, log_variance)
print(f"generated:\n {generated}")
decoded = compression_VAE.decode(generated)
print("decoded")
print(decoded[1])

# compression_VAE.train(np.array([np.arange(5)]), 10, 10)
# compressed_sounds = np.asarray([compression_VAE.encode(sound) for sound in sounds])

