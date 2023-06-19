from network_tester import test_vae, test_base
import audio_parsing.audio_parsing as audio
import cProfile
import pstats
import subprocess
import numpy as np

sounds, names = audio.get_raw_data(32, 500)
# print(sounds)
# sounds, names = audio.get_raw_data(32, 5)

# This is just a way to test the sound data
# audio.play_random_sound(sounds, names)

song_length = len(sounds[0])
print(f"Current Number of Songs: {len(sounds)}")
print(f"Current Size of Each Song: {len(sounds[0])}")

# with cProfile.Profile() as profile:
# test_vae(sounds, .1, 10000, (song_length, 30, 25), decoder_layers=(25, 29, song_length), loss_graph=True, test_data=False)

# results = pstats.Stats(profile)
# results.sort_stats(pstats.SortKey.CUMULATIVE)
# results.dump_stats("profile.prof")

# subprocess.run(["snakeviz", "profile.prof"]) 

test_base(np.stack((sounds,sounds), axis=1), .1, 1000, (song_length, 300, 100, 90, 300, 100, song_length), loss_graph=True, test_data=False)
# test_base(np.stack((sounds,sounds), axis=1), .1, 100, (song_length, 5, 3, 4, song_length), loss_graph=True, test_data=False)