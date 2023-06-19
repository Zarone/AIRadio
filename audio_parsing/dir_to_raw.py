import librosa
import numpy as np

SAMPLING_RATE: float = 30000
SCALE_FACTOR = 1E1
def get_raw_audio(location: str, max_elements) -> np.ndarray:
  # Librosa returns a different number of samples, so this sets an acceptable upper bound
  END_POINT=30*SAMPLING_RATE-SAMPLING_RATE//10

  try:
    audio_data, _ = librosa.load(location, sr=SAMPLING_RATE)
  except:
    raise Exception(f"ERROR IN FILE {location}")
  audio = audio_data[0:END_POINT].reshape(-1, 1)[0:max_elements]
  
  # epsilon_naught is to prevent divide by zero error
  epsilon_naught = 1E-15
  mean = np.mean(audio)
  current_std = np.std(audio) + epsilon_naught 

  scaled_audio = (audio - mean) * (SCALE_FACTOR/current_std)

  return scaled_audio

