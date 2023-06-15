import numpy as np
import audio_parsing.audio_parsing as audio
from neural_networks.vae.normal_vae.normal_vae import VAE

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
 
compression_VAE = VAE(encoder_layers, decoder_layers)
compression_VAE.train(sounds, 5000, len(sounds), graph=True, learning_rate=0.05)

print(f"test input:\n {sounds[0]*song_length}")

_, _, mu, log_variance = compression_VAE.encode( sounds[0] )

# print(f"mu:\n {mu} \nsigma:\n {np.exp(0.5*log_variance)}")

generated, epsilon = compression_VAE.gen(mu, -100)

# print(f"generated:\n {generated}")

decoded = compression_VAE.decode(generated)[1]

print(f"decoded:\n {decoded[-1]*song_length}")

print(f"off by:\n {song_length*(decoded[-1]-sounds[0])}")

# compressed_sounds = np.asarray([compression_VAE.encode(sound) for sound in sounds])

