import numpy as np

HEADING_GREEN= "\033[1;32m"
HEADING_RED = "\033[1;31m"
GREEN = "\033[0;32m"
RED = "\033[0;31m"
END_FORMAT = "\033[0;0m"
INDENT = " - "

class Tester:
  def __init__(self, name: str):
    self.name = name
    self.passed_all = True
    self.tests = []

  def __del__(self):
    for test in self.tests:
      if not test[1]: self.passed_all = False

    if self.passed_all:
      print(f"{HEADING_GREEN}{self.name} Passed{END_FORMAT}")
    else:
      print(f"{HEADING_RED}{self.name} Failed{END_FORMAT}")

    for test in self.tests:
      self.unit_test(test[0], test[1])

  def tester(self, name: str, passed: bool):
    self.tests.append((name, passed)) 

  @staticmethod
  def eq(x, y, err: float=0.001):
    if isinstance(x, float) or isinstance(x, int):
      return Tester.num_equal(x,y,err)
    elif isinstance(x, np.ndarray):
      return Tester.arr_equal(x,y,err)
    else:
      print(f"Unrecognized Type: {type(x)}")


  @staticmethod
  def num_equal(x: float, y: float, err: float=0.001):
    return abs(x-y)<err

  @staticmethod
  def arr_equal(x: np.ndarray, y: np.ndarray, err: float=0.001):
    for i, _ in enumerate(x):
      if not Tester.num_equal(x[i], y[i]):
        return False
    return True

  def unit_test(self, name: str, passed: bool):
    if passed:
      print(f"{GREEN}{INDENT}Test {name} Passed{END_FORMAT}")
    else:
      print(f"{RED}{INDENT}Test {name} Failed{END_FORMAT}")

