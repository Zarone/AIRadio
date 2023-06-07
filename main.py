import audio_parsing.audio_parsing as audio
from neural_networks.vae.normal_vae.normal_vae import VAE

sounds, names = audio.get_raw_data()

# This is just a way to test the sound data
# audio.play_random_sound(sounds, names)

print(f"Current Size of Each Song: {len(sounds[0])}")

# TODO

compression_VAE = VAE([10, 5, 1])
# compression_VAE.train()
# compressed_sounds = np.asarray([compression_VAE.encode(sound) for sound in sounds])

