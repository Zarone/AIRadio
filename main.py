import audio_parsing.audio_parsing as audio
from neural_networks.components.base import BaseNetwork
from neural_networks.vae.normal_vae.normal_vae import VAE
from neural_networks.vae.recurrent_vae.recurrent_vae import RecurrentVAE
from neural_networks.components.optimizer.adam import Adam

AMPLITUDE_SCALE = 1
NUM_AMPLITUDES = 6
NUM_FILES = 5

sounds, names = audio.get_raw_data(NUM_FILES, NUM_AMPLITUDES, AMPLITUDE_SCALE)

"""
# This is just a way to test the sound data
song = audio.play_random_sound(sounds, names, AMPLITUDE_SCALE)
"""


"""
train_compressor(sounds, COMPRESSION_1_INFO, 10000)

compressed, ae1 = compress(song, COMPRESSION_1_INFO)
print(f"Current Size of Song: {len(compressed)}")

decompressed = decompress(ae1, compressed, COMPRESSION_1_INFO)
print(f"Current Size of Song: {len(decompressed)}")

audio.play_audio(decompressed, AMPLITUDE_SCALE)

audio.plot_audio_comparison(song, decompressed)
"""


"""
# network = BaseNetwork(
    # layers=(5, 4, 3, 3, 3, 4, 5),
    # optimizer=Adam(loss_taperoff=True)
# )
# formatted_sounds = network.format_unsupervised_input(sounds)

# network.train(
    # formatted_sounds,
    # batch_size=5,
    # max_epochs=20000,
    # graph=True,
    # learning_rate=0.01
# )

# network: VAE = VAE(encoder_layers=(5, 4, 3, 3), decoder_layers=(3, 3, 4, 5))
# network.train(
    # sounds,
    # batch_size=5,
    # max_epochs=20000,
    # graph=True,
    # learning_rate=0.01
# )
"""


from neural_networks.components.activations import leaky_relu, leaky_relu_derivative, relu, relu_derivative, sigmoid, sigmoid_derivative
network: RecurrentVAE = RecurrentVAE(
    (3, 4), (4, 3),
    latent_recurrent_layers=(4, 4),
    output_recurrent_layers=(3, 3),
    optimizer=Adam(loss_taperoff=True),
    activation=relu,
    activation_derivative=relu_derivative
)
time_separated_sounds = network.get_time_seperated_data(sounds)
network.train(
    time_separated_sounds,
    batch_size=5,
    max_epochs=20000,
    graph=True,
    learning_rate=1e-4
)
# train_compressor(sounds, COMPRESSION_1_INFO, 10000)

# compressed, ae1 = compress(song, COMPRESSION_1_INFO)
# print(f"Current Size of Song: {len(compressed)}")

# decompressed = decompress(ae1, compressed, COMPRESSION_1_INFO)
# print(f"Current Size of Song: {len(decompressed)}")

# audio.play_audio(decompressed, AMPLITUDE_SCALE)

# audio.plot_audio_comparison(song, decompressed)

print("time_separated_sounds[0]")
print(time_separated_sounds[0])
print("network.feedforward(time_separated_sounds[0])")
print(network.feedforward(time_separated_sounds[0]))
