from tools.network_tester import *
import audio_parsing.audio_parsing as audio
from tools.profile import ProfileWrapper
from compression.compression import compress, COMPRESSION_1_INFO, COMPRESSION_2_INFO, decompress, train_compressor
from neural_networks.vae.recurrent_vae.recurrent_vae import RecurrentVAE

AMPLITUDE_SCALE = 10

MAX_AMPLITUDES = 899000
# MAX_AMPLITUDES = 1000

sounds, names = audio.get_raw_data(100, MAX_AMPLITUDES, AMPLITUDE_SCALE)

# This is just a way to test the sound data
song = audio.play_random_sound(sounds, names, AMPLITUDE_SCALE)

song_length = len(sounds[0])
print(f"Current Number of Songs: {len(sounds)}")
print(f"Current Size of Song: {len(song)}")

# train_compressor(sounds, COMPRESSION_1_INFO, 1000)

compressed, ae1 = compress(song, COMPRESSION_1_INFO)
print(f"Current Size of Song: {len(compressed)}")

  # compressed, ae2 = compress(compressed, COMPRESSION_2_INFO)
  # print(f"Current Size of Song 0: {len(compressed)}")

# AMPLITUDES_PER_ITER = 5
# ITERATIONS = len(compressed)//AMPLITUDES_PER_ITER

# print("AMPLITUDES_PER_ITER", AMPLITUDES_PER_ITER)
# print("ITERATIONS", ITERATIONS)

  # decompressed = decompress(ae2, compressed, COMPRESSION_2_INFO)
  # print(f"Current Size of Song 0: {len(decompressed)}")

decompressed = decompress(ae1, compressed, COMPRESSION_1_INFO)
print(f"Current Size of Song: {len(decompressed)}")

audio.play_audio(decompressed, AMPLITUDE_SCALE)
# print("song", song)
# print("decompressed", decompressed)

audio.plot_audio_comparison(song, decompressed)

# network = RecurrentVAE((5, 4, 3, 3), (3, 3, 4, 5))
# network.train(sounds)
