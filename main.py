from neural_networks.autoencoder.autoencoder import AutoEncoder
from tools.network_tester import *
import audio_parsing.audio_parsing as audio
import numpy as np
from tools.profile import ProfileWrapper

MAX = 899000
INIT_CHUNK_SIZE = 10
CHUNKS = MAX//INIT_CHUNK_SIZE
COMPRESSION_1_CHUNK_SIZE = 1
sounds, names = audio.get_raw_data(20, 10)

# This is just a way to test the sound data
# audio.play_random_sound(sounds, names)

song_length = len(sounds[0])
print(f"Current Number of Songs: {len(sounds)}")
print(f"Current Size of Each Song: {len(sounds[0])}")

layers = (song_length, 40, 20, COMPRESSION_1_CHUNK_SIZE, 20, 40, song_length)

ae = AutoEncoder(layers)
ae.train(sounds, max_epochs=100, batch_size=20, learning_rate=0.005)
# ae.init_from_file("CompressionParameters.npz")

single_compressed_chunks = np.zeros((CHUNKS*COMPRESSION_1_CHUNK_SIZE, 1))
print("single_compressed_chunks", single_compressed_chunks)
for i in range(CHUNKS):
  print("section", single_compressed_chunks[i*COMPRESSION_1_CHUNK_SIZE:(i+1)*COMPRESSION_1_CHUNK_SIZE])
  print("encoded sections", ae.encode(sounds[0])[1][-1])
  single_compressed_chunks[i*COMPRESSION_1_CHUNK_SIZE:(i+1)*COMPRESSION_1_CHUNK_SIZE] = ae.encode(sounds[0])[1][-1]

# network.save_to_file("CompressionParameters.npz")