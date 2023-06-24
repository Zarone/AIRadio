from typing import Dict, List
import numpy as np
from neural_networks.autoencoder.autoencoder import AutoEncoder

COMPRESSION_1_INFO = {
      "previous_chunk_size": 100,
      "chunk_size": 45,
      "layers": (100, 60, 45, 60, 100),
      "file": "Compression1Parameters.npz"
    }

def train_compressor(sounds: List, details: Dict, training_data_count: int):
  N = len(sounds)
  full_song_size = len(sounds[0])

  array_indices = np.random.randint(0, N, training_data_count)  # Randomly select N array indices
  start_indices = np.random.randint(0, full_song_size - details["previous_chunk_size"], training_data_count)  # Random start indices

  sequences = [ \
      sounds[ array_indices[i] ][ start_indices[i]: start_indices[i]+details["previous_chunk_size"] ] for i in range(training_data_count) \
      ]

  ae = AutoEncoder(details["layers"])
  ae.train(sequences, max_epochs=1000, batch_size=min(2500, training_data_count), learning_rate=0.05)
  ae.save_to_file(details["file"])

def compress(audio: np.ndarray, details: Dict):
  chunks = len(audio)//details["previous_chunk_size"]
  ae = AutoEncoder(details["layers"])
  ae.init_from_file(details["file"])

  compressed_chunks = np.zeros( (chunks*details["chunk_size"], 1) )
  for i in range(chunks):
    compressed_chunks[ i*details["chunk_size"]:(i+1)*details["chunk_size"] ] = \
      ae.encode(
          audio[ i*details["previous_chunk_size"] : (i+1)*details["previous_chunk_size"] ]
        )[1][-1]
  return compressed_chunks, ae

def decompress(ae: AutoEncoder, compressed, details):
  chunks = len(compressed)//details["chunk_size"]
  decompressed_chunks = np.zeros( (chunks*details["previous_chunk_size"], 1) )
  for i in range(chunks):
    decompressed_chunks[ i*details["previous_chunk_size"]:(i+1)*details["previous_chunk_size"] ] = \
      ae.decode(
          compressed[ i*details["chunk_size"] : (i+1)*details["chunk_size"] ]
        )[1][-1]

  return decompressed_chunks
