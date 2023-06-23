import matplotlib.pyplot as plt
from typing import List, Tuple, Any
from audio_parsing.dir_to_raw import get_raw_audio, SAMPLING_RATE
from audio_parsing.get_sounds import get_directories
import sounddevice as sd
import numpy as np
import numpy.typing as npt
import random

def get_raw_data(num_files, data_points, scale: float) -> Tuple[List, List]:
  dir = get_directories(num_files)
  song_names = [song_dir for song_dir in dir]
  print(f"Getting {len(dir)} songs...")
  song_data = ([get_raw_audio(song_dir, data_points, scale) for song_dir in dir])
  # song_data = np.array([get_raw_audio(song_dir, data_points) for song_dir in dir])
  return (song_data, song_names)

def play_audio(audio_data, scale: float) -> None:
  sd.play(audio_data/scale/10, SAMPLING_RATE)
  sd.wait()  # Wait until the audio finishes playing

# Pick a random song and play it
def play_random_sound(sounds: List, names: List[str], scale: float):
  choice = random.randint(0, len(names)-1)
  print(f"Playing song: {names[choice]}")
  # play_audio(sounds[choice], scale)
  return sounds[choice]

def plot_audio_comparison(audio_real, audio_gen):
  CUT = 100

  audio1 = audio_real[ 0 : len(audio_real)//CUT ]
  audio2 = audio_gen [ 0 : len(audio_gen )//CUT ]

  _, axs = plt.subplots(2, sharex=True)

  axs[0].plot(audio1, "blue", label="Real Audio")
  axs[0].grid()
  axs[1].plot(audio2, "red", label="Generated Audio")
  axs[1].grid()

  plt.show()

