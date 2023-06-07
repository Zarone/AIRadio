import audio_parsing.audio_parsing as audio

sounds, names = audio.get_raw_data()

# This is just a way to test the sound data
# audio.play_random_sound(sounds, names)

print(f"Current Size of Each Song: {len(sounds[0])}")

# TODO

# compression_VAE = new VAE(layers)
# compression_VAE.train()
# compressed_sounds = np.asarray([compression_VAE.encode(sound) for sound in sounds])

