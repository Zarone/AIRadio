from network_tester import test_network
import audio_parsing.audio_parsing as audio
import cProfile
import pstats
import subprocess

sounds, names = audio.get_raw_data()

# This is just a way to test the sound data
# audio.play_random_sound(sounds, names)

song_length = len(sounds[0])
print(f"Current Number of Songs: {len(sounds)}")
print(f"Current Size of Each Song: {len(sounds[0])}")

with cProfile.Profile() as profile:
  test_network(sounds, .1, 10000, (song_length, 30, 25), decoder_layers=(25, 29, song_length))

results = pstats.Stats(profile)
results.sort_stats(pstats.SortKey.CUMULATIVE)
results.dump_stats("profile.prof")

subprocess.run(["snakeviz", "profile.prof"]) 

