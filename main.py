import audio_parsing.audio_parsing as audio

AMPLITUDE_SCALE = 1
NUM_AMPLITUDES = 15
NUM_FILES = 5

sounds, names = audio.get_raw_data(NUM_FILES, NUM_AMPLITUDES, AMPLITUDE_SCALE)

"""
# This is just a way to test the sound data
song = audio.play_random_sound(sounds, names, AMPLITUDE_SCALE)
"""
