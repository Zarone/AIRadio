from network_tester import test_network
import audio_parsing.audio_parsing as audio

sounds, names = audio.get_raw_data()

# This is just a way to test the sound data
# audio.play_random_sound(sounds, names)

song_length = len(sounds[0])
print(f"Current Number of Songs: {len(sounds)}")
print(f"Current Size of Each Song: {len(sounds[0])}")
test_network(sounds, .1, 200, (song_length, 30, 25))
