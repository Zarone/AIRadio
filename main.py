import audio_parsing.audio_parsing as audio
from neural_networks.components.base import BaseNetwork

AMPLITUDE_SCALE = 1
NUM_AMPLITUDES = 5
NUM_FILES = 6

sounds, names = audio.get_raw_data(NUM_FILES, NUM_AMPLITUDES, AMPLITUDE_SCALE)

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

network = BaseNetwork(layers=(5, 4, 3, 3, 3, 4, 5))
network.train(
    # sounds,
    network.format_unsupervised_input(sounds),
    batch_size=6,
    max_epochs=5000,
    graph=True,
    learning_rate=0.05
)

# network: VAE = VAE(encoder_layers=(5, 3), decoder_layers=(3, 5))
# network.train(
    # sounds,
    # batch_size=5,
    # max_epochs=5000,
    # graph=True,
    # learning_rate=0.05
# )

# network: RecurrentVAE = RecurrentVAE((5, 4, 3, 3), (3, 3, 4, 5))
# time_seperated_sounds: np.ndarray = network.get_time_seperated_data(sounds)
# network.train(
    # time_seperated_sounds,
    # batch_size=5,
    # max_epochs=5000,
    # graph=True,
    # learning_rate=0.05
# )
