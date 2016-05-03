from abc import abstractmethod

class Solver:
    def __init__(self, test=None, full=True, L=True, OD=True, CP=True, LP=True, eq=None,
                 init=False, noise=0, method=None):
        self.test = test  # filename in DATA_DIR to load
        self.eq = eq  # Equality constraint, 'CP' or 'OD'
        self.init = init  # Toggle to compute init solution via lsqr

        self.full = full  # Toggle using all links (vs observed links)
        self.L = L  # Toggle using link sensor info
        self.OD = OD  # Toggle using OD sensor info
        self.CP = CP  # Toggle using CP sensor info
        self.LP = LP  # Toggle using LP sensor info
        self.noise = noise  # Additive Gaussian noise factor
        self.method = method  # Method, specific to solver type

        self.output = None

    @abstractmethod
    def setup(self, data):
        return NotImplemented

    @abstractmethod
    def solve(self):
        return NotImplemented

    @abstractmethod
    def analyze(self):
        return NotImplemented