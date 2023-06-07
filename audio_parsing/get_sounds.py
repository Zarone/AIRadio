from typing import List
from audio_parsing.dir_to_raw import get_raw_audio
import glob 

TEST_AUDIO_FILE = './music_sample/000/000002.mp3'
audio_files = [ TEST_AUDIO_FILE ]

def get_directories() -> List[str]:
  return glob.glob("./music_sample/*/*.mp3")[:30]