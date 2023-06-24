import librosa
import numpy as np

SAMPLING_RATE: float = 30000

i = 0

def get_raw_audio(location: str, max_elements: int, scale: float) -> np.ndarray:
  try:
    audio_data, _ = librosa.load(location, sr=SAMPLING_RATE)
  except:
    raise Exception(f"ERROR IN FILE {location}")
  audio = audio_data[0:max_elements].reshape(-1, 1)
  
  # epsilon_naught is to prevent divide by zero error
  epsilon_naught = 1E-15
  mean = np.mean(audio)
  current_std = np.std(audio) + epsilon_naught 

  scaled_audio = (audio - mean) * (scale/current_std)
  global i
  print(f"Loaded audio from {location}, element {i}")

  if not len(scaled_audio) == max_elements:
    raise Exception(f"element {i} from location {location} failed to load")

  i+=1

  return scaled_audio

