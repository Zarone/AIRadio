import numpy as np
import audio_parsing.audio_parsing as audio
from neural_networks.vae.normal_vae.normal_vae import VAE

sounds, names = audio.get_raw_data()

# This is just a way to test the sound data
# audio.play_random_sound(sounds, names)

song_length = len(sounds[0])
print(f"Current Size of Each Song: {len(sounds[0])}")

encoder_layers = (song_length, 20, 10)
decoder_layers = encoder_layers[::-1]

compression_VAE = VAE(encoder_layers, decoder_layers)

# test_input1 = np.arange(5).reshape(-1,1)
# test_input2 = np.arange(1, 10, 2).reshape(-1,1)
# test_input3 = np.arange(1, 20, 4).reshape(-1,1)
# test_input4 = np.arange(1, 15, 3).reshape(-1,1)
# test_input = np.array([test_input1, test_input2, test_input3, test_input4])

compression_VAE.train(sounds[0:11], 10, 5, graph=True, learning_rate=0.001)

print(f"test input:\n {sounds[0]}")

_, _, mu, log_variance = compression_VAE.encode( sounds[0] )

print(f"mu:\n {mu} \nsigma:\n {np.exp(0.5*log_variance)}")

generated, epsilon = compression_VAE.gen(mu, log_variance)

print(f"generated:\n {generated}")

decoded = compression_VAE.decode(generated)[1]

print(f"decoded:\n {decoded}")

# compressed_sounds = np.asarray([compression_VAE.encode(sound) for sound in sounds])

