import audio_parsing.audio_parsing as audio
from neural_networks.components.base import BaseNetwork
from neural_networks.vae.vae import VAE
from neural_networks.components.optimizer.adam_w_taperoff import AdamTaperoff

AMPLITUDE_SCALE = 1
NUM_AMPLITUDES = 5
NUM_FILES = 1

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
network = BaseNetwork(
    layers=(5, 4, 5),
    optimizer=AdamTaperoff()
)

formatted_sounds = network.format_unsupervised_input(sounds)

network.train(
    formatted_sounds,
    batch_size=10,
    max_epochs=10000,
    graph=True,
    learning_rate=0.01
)
"""

network: VAE = VAE(encoder_layers=(5, 4, 3, 3), decoder_layers=(3, 3, 4, 5))
network.train(
    sounds,
    batch_size=1,
    max_epochs=20000,
    graph=True,
    learning_rate=0.01
)


"""
network: RecurrentVAE = RecurrentVAE(
    (3, 100), (100, 3),
    latent_recurrent_layers=(100, 100),
    output_recurrent_layers=(3, 3),
    optimizer=Adam(loss_taperoff=True),
    activation=relu,
    activation_derivative=relu_derivative
)
time_separated_sounds = network.get_time_seperated_data(sounds)
network.train(
    time_separated_sounds,
    batch_size=1,
    max_epochs=1000,
    graph=True,
    learning_rate=1e-15
)

print("time_separated_sounds[0]")
print(time_separated_sounds[0])
print("network.feedforward(time_separated_sounds[0])")
print(network.feedforward(time_separated_sounds[0]))
"""
