import time

import config as c
from Solver import Solver
from BSC_NNLS.python.util import load_raw
from scipy.sparse.linalg import lsqr

# FIXME temporary hack
from scenario_utils import LS_postprocess

class SolverLSQR(Solver):
    def __init__(self, test=None, data=None, full=True, L=True, OD=True,
                  CP=True, LP=True, eq='CP', damp=0):
        Solver.__init__(self)

        self.test = test
        self.eq = eq
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
        # Read from data dict based on scenario parameters
        AA, bb, x_true, T, d, U, f, nz, output = load_raw(self.data,
                            full=self.full, L=self.L, OD=self.OD, CP=self.CP,
                            LP=self.LP, solve=True, damp=self.damp)
        # Issue lsqr solver
        if AA is None:
            output['error'] = "AA,bb is empty"
            self.A,self.b,self.x0,self.x_true,out = None,None,None,None,output
        else:
            x, istop, itn, r1norm, r2norm, anorm, acond, arnorm, xnorm, var = \
                lsqr(AA,bb,damp=self.damp)
            output['istop'], output['init_iters'], output['r1norm'],\
            output['r2norm'], output['anorm'], output['acond'], \
            output['arnorm'],output['xnorm'] = \
                istop, itn, r1norm, r2norm, anorm, acond, arnorm, xnorm

            self.A, self.b, self.x0, self.x_true, out = AA, bb, x, x_true, output

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
    from comparison.tests.test_solver_lsqr import TestSolverLSQR
    unittest.main()
