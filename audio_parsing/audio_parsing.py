from typing import List, Tuple
from audio_parsing.dir_to_raw import get_raw_audio, SAMPLING_RATE
from audio_parsing.get_sounds import get_directories
import sounddevice as sd
import numpy as np

def get_raw_data() -> Tuple[np.array, List]:
  song_names = [song_dir for song_dir in get_directories()]
  song_data = np.asarray([get_raw_audio(song_dir) for song_dir in get_directories()])
  return (song_data, song_names)

def play_audio(audio_data) -> None:
  sd.play(audio_data, SAMPLING_RATE)
  sd.wait()  # Wait until the audio finishes playing