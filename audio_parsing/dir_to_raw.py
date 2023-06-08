import librosa
import numpy as np

SAMPLING_RATE: float = 30000
SCALE_FACTOR = 1E11
def get_raw_audio(location: str) -> np.ndarray:
  # Librosa returns a different number of samples, so this sets an acceptable upper bound
  END_POINT=30*SAMPLING_RATE-SAMPLING_RATE//10

  return librosa.load(location, sr=SAMPLING_RATE)[0][0:END_POINT].reshape(-1, 1)*SCALE_FACTOR

