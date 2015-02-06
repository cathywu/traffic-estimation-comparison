from abc import abstractmethod

class Solver:
    def __init__(self):
        self.output = None

    @abstractmethod
    def setup(self):
        return NotImplemented

    @abstractmethod
    def solve(self):
        return NotImplemented

    @abstractmethod
    def analyze(self):
        return NotImplemented