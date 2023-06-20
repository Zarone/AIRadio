import cProfile
import pstats
import subprocess

class ProfileWrapper:
  def __enter__(self):
    self.profile = cProfile.Profile()
    self.profile.enable()
  def __exit__(self, exc_type, exc_value, exc_tb):
    self.profile.disable()
    results = pstats.Stats(self.profile)
    results.sort_stats(pstats.SortKey.CUMULATIVE)
    results.dump_stats("profile.prof")
    subprocess.run(["snakeviz", "profile.prof"]) 