import time
import logging

import numpy as np

from Solver import Solver
from python.util import solver_input, load_data

# FIXME temporary hack
from scenario_utils import LS_postprocess

class SolverLSQR(Solver):
    def __init__(self, args, test=None, data=None, full=True, L=True, OD=True,
                  CP=True, LP=True, eq='CP', init=True, damp=0):
        Solver.__init__(self)

        self.args = args
        self.test = test
        self.eq = eq
        self.init = init
        self.full = full
        self.L = L
        self.OD = OD
        self.CP = CP
        self.LP = LP
        self.damp = damp

    def setup(self,data):
        self.data = data

    def solve(self):
        init_time = time.time()
        self.A, self.b, self.x0, self.x_true, out = solver_input(self.data, full=self.full, L=self.L, OD=self.OD, CP=self.CP,
                                             LP=self.LP, solve=True, damp=self.damp)
        init_time = time.time() - init_time
        self.output = out
        self.output['duration'] = init_time

        if self.A is None:
            self.output['error'] = "Empty objective"
            return self.output

    def analyze(self):
        x_last, error, self.output = LS_postprocess([self.x0],self.x0,self.A,self.b,
                                               self.x_true,output=self.output,is_x=True)
        self.output['iters'], self.output['times'] = [0], [0]

if __name__ == "__main__":
    import unittest
    from tests.test_solver_lsqr import TestSolverLSQR
    unittest.main()
