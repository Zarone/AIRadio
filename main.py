from tools.network_tester import *
import audio_parsing.audio_parsing as audio
from compression.compression import compress, COMPRESSION_1_INFO, decompress, train_compressor
from neural_networks.vae.recurrent_vae.recurrent_vae import RecurrentVAE
import numpy as np
import typing

AMPLITUDE_SCALE = 10

# MAX_AMPLITUDES = -1
MAX_AMPLITUDES = 10

# sounds, names = audio.get_raw_data(5, MAX_AMPLITUDES, AMPLITUDE_SCALE)

# # This is just a way to test the sound data
# song = audio.play_random_sound(sounds, names, AMPLITUDE_SCALE)

# song_length = len(sounds[0])
# print(f"Current Number of Songs: {len(sounds)}")
# print(f"Current Size of Song: {len(song)}")

# train_compressor(sounds, COMPRESSION_1_INFO, 10000)

# compressed, ae1 = compress(song, COMPRESSION_1_INFO)
# print(f"Current Size of Song: {len(compressed)}")

# decompressed = decompress(ae1, compressed, COMPRESSION_1_INFO)
# print(f"Current Size of Song: {len(decompressed)}")

# audio.play_audio(decompressed, AMPLITUDE_SCALE)

# audio.plot_audio_comparison(song, decompressed)

# network = RecurrentVAE((5, 4, 3, 3), (3, 3, 4, 5))
# time_seperated_sounds: np.ndarray = network.get_time_seperated_data(sounds)
# encoded = network.encode(time_seperated_sounds[0])
# print("encoded[0]", encoded[0])
# print("encoded[1]", encoded[1])
# print("encoded[2]", encoded[2])
# print("encoded[3]", encoded[3])
# print("encoded[4]", encoded[4])
# network.train(sounds)
