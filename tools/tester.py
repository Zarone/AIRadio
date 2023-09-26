import numpy as np
from typing import List, Tuple

HEADING_GREEN = "\033[1;32m"
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
        self.last_x = 0
        self.last_y = 0
        self.test_data = []

    def __enter__(self):
        return self

    def __exit__(self, *_):
        for test in self.tests:
            if not test[1]:
                self.passed_all = False

        if self.passed_all:
            print(f"{HEADING_GREEN}{self.name} Passed{END_FORMAT}")
        else:
            print(f"{HEADING_RED}{self.name} Failed{END_FORMAT}")

        for i, test in enumerate(self.tests):
            self.unit_test(test[0], test[1], i)

        assert self.passed_all

    def tester(self, name: str, passed: bool):
        self.tests.append((name, passed))
        self.test_data.append((self.last_x, self.last_y))

    def eq(self, x, y, err: float = 0.001, init_call=True):
        if (
            isinstance(x, (float, int, np.integer, np.bool_)) and
            isinstance(y, (np.ndarray, List, Tuple, dict))
        ):
            return False
        if (
            isinstance(y, (float, int, np.integer, np.bool_)) and
            isinstance(x, (np.ndarray, List, Tuple, dict))
        ):
            return False

        if init_call:
            self.last_x = x
            self.last_y = y
        if isinstance(x, (float, int, np.integer, np.bool_)):
            return self.num_equal(x, y, err)
        elif isinstance(x, (np.ndarray, List, Tuple)):
            return self.arr_equal(x, y, err)
        elif isinstance(x, dict):
            return self.dict_equal(x, y, err)
        else:
            raise Exception(f"Unrecognized Type: {type(x)}")

    def num_equal(
        self,
        x: float | int | np.integer | np.bool_,
        y: float | int | np.integer,
        err: float = 0.001
    ):
        if not abs(x-y) < err:
            print(f"Expected {y}, got {x}")
        return abs(x-y) < err

    def dict_equal(
        self,
        x,
        y,
        err
    ):
        if len(x) != len(y):
            return False
        for key, val in x.items():
            if key not in y or not self.eq(x[key], y[key], err, False):
                return False
        return True

    def arr_equal(
        self,
        x: np.ndarray | List | Tuple,
        y: np.ndarray | List | Tuple,
        err: float = 0.001
    ):
        if len(x) != len(y):
            return False
        for i, _ in enumerate(x):
            if not self.eq(x[i], y[i], err, False):
                return False
        return True

    def unit_test(self, name: str, passed: bool, index: int):
        if passed:
            print(f"{GREEN}{INDENT}Test {name} Passed{END_FORMAT}")
        else:
            print(f"{RED}{INDENT}Test {name} Failed{END_FORMAT}")
            print(
                f"{RED}  {INDENT}Expected {self.test_data[index][1]}, Got {self.test_data[index][0]}")
