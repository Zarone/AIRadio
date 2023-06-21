from tools.network_tester import test_vae, test_base
import audio_parsing.audio_parsing as audio
import numpy as np

sounds, names = audio.get_raw_data(32, 500)

# This is just a way to test the sound data
# audio.play_random_sound(sounds, names)

song_length = len(sounds[0])
print(f"Current Number of Songs: {len(sounds)}")
print(f"Current Size of Each Song: {len(sounds[0])}")

input_output_sounds = np.stack((sounds,sounds), axis=1)
# with ProfileWrapper():

# test_vae(sounds, .1, 1000, (song_length, 30, 10, 9), decoder_layers=(9, 30, 10, song_length), loss_graph=False, test_data=False)

# test_base(input_output_sounds, .1, 1000, (song_length, 30, 10, 9, 30, 10, song_length), loss_graph=False, test_data=False)
