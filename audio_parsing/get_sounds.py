from typing import List
import glob 

TEST_AUDIO_FILE = './music_sample/000/000002.mp3'
audio_files = [ TEST_AUDIO_FILE ]

# Gets a list of all filenames of songs in the dataset
def get_directories(num_files: int) -> List[str]:
  return glob.glob("./music_sample/*/*.mp3")[:num_files]
