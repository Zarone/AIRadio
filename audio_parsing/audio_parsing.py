from typing import List, Tuple
from audio_parsing.dir_to_raw import get_raw_audio, SAMPLING_RATE
from audio_parsing.get_sounds import get_directories
import sounddevice as sd
import numpy as np
import random

def get_raw_data() -> Tuple[np.ndarray, List]:
  dir = get_directories()
  song_names = [song_dir for song_dir in dir]
  max_elements = 30
  print(f"REMINDER: I shrunk that song size to {max_elements} element")
  song_data = np.array([get_raw_audio(song_dir)[0:max_elements] for song_dir in dir])
  return (song_data, song_names)

def play_audio(audio_data) -> None:
  sd.play(audio_data, SAMPLING_RATE)
  sd.wait()  # Wait until the audio finishes playing

# Pick a random song and play it
def play_random_sound(sounds: np.ndarray, names: List[str]):
  choice = random.randint(0, len(names)-1)
  print(f"Playing song: {names[choice]}")
  play_audio(sounds[choice])
