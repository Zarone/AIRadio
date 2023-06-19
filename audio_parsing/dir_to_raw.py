import librosa
import numpy as np

SAMPLING_RATE: float = 30000
SCALE_FACTOR = 1E1
MAX_ELELENTS = 30
def get_raw_audio(location: str) -> np.ndarray:
  # Librosa returns a different number of samples, so this sets an acceptable upper bound
  END_POINT=30*SAMPLING_RATE-SAMPLING_RATE//10

  try:
    audio_data, _ = librosa.load(location, sr=SAMPLING_RATE)
  except:
    raise Exception(f"ERROR IN FILE {location}")
  audio = audio_data[0:END_POINT].reshape(-1, 1)[0:MAX_ELELENTS]
  
  # epsilon_naught is to prevent divide by zero error
  epsilon_naught = 1E-15
  maximum = np.max(np.abs(audio)) + epsilon_naught

  return audio/maximum*SCALE_FACTOR

