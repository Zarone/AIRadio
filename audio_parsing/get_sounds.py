from typing import List
import glob 

TEST_AUDIO_FILE = './music_sample/000/000002.mp3'
audio_files = [ TEST_AUDIO_FILE ]

# Gets a list of all filenames of songs in the dataset
SONG_COUNT: int = 16
def get_directories() -> List[str]:
  return glob.glob("./music_sample/*/*.mp3")[:SONG_COUNT]
