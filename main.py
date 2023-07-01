from neural_networks.components.optimizer.optimizer import Optimizer
from neural_networks.components.optimizer.SGD import SGD
from tools.network_tester import *
import audio_parsing.audio_parsing as audio
from compression.compression import compress, COMPRESSION_1_INFO, decompress, train_compressor
from neural_networks.vae.recurrent_vae.recurrent_vae import RecurrentVAE
import numpy as np
import typing

AMPLITUDE_SCALE = 1

# MAX_AMPLITUDES = -1
MAX_AMPLITUDES = 50

sounds, names = audio.get_raw_data(5, MAX_AMPLITUDES, AMPLITUDE_SCALE)

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


network: RecurrentVAE = RecurrentVAE((5, 4, 3, 3), (3, 3, 4, 5))
time_seperated_sounds: np.ndarray = network.get_time_seperated_data(sounds)
network.train(time_seperated_sounds, batch_size=5, max_epochs=5000, graph=True, learning_rate=0.001)
