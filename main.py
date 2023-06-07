import audio_parsing.audio_parsing as audio
import random

sounds, names = audio.get_raw_data()

choice = random.randint(0, len(names))
print(f"Playing song: {names[choice]}")
audio.play_audio(sounds[choice])