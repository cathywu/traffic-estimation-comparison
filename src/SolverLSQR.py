import time

import numpy as np

import config as c
from Solver import Solver
from BSLS.python.util import load_raw
from scipy.sparse.linalg import lsqr

# FIXME temporary hack
from scenario_utils import LS_postprocess
from BSLS.python.bsls_matrices import BSLSMatrices


class SolverLSQR(Solver):
    def __init__(self, test=None, full=True, L=True, OD=True, CP=True, LP=True,
                 eq='CP', damp=0, noise=0.0):
        Solver.__init__(self)

        self.test = test
        self.eq = eq
        self.full = full
        self.L = L
        self.OD = OD
        self.CP = CP
        self.LP = LP
        self.damp = damp

        self.data, self.A, self.b, self.x0, self.x_true = None, None, None, \
            None, None

    def setup(self, data):
        self.data = data
        init_time = time.time()
        # Read from data dict based on scenario parameters
        bm = BSLSMatrices(data=self.data, full=self.full, L=self.L, OD=self.OD,
                          CP=self.CP, LP=self.LP)
        A, b, x_true = bm.get_LSQR()
        self.output = bm.info
        init_time = time.time() - init_time
        self.output['init_time'] = init_time
        self.A, self.b, self.x_true = A, b, x_true

        if self.noise:
            b_true = self.b
            delta = np.random.normal(scale=self.b*self.noise)
            self.b = self.b + delta

    def solve(self):
        solve_time = time.time()
        # Issue lsqr solver
        if self.A is None:
            self.output['error'] = "AA,bb is empty"
            self.A, self.b, self.x0, self.x_true = None, None, None, None
        else:
            x0, istop, itn, r1norm, r2norm, anorm, acond, arnorm, xnorm, var = \
                lsqr(self.A, self.b, damp=self.damp)
            output = self.output
            output['istop'], output['init_iters'], output['r1norm'],\
                output['r2norm'], output['anorm'], output['acond'], \
                output['arnorm'], output['xnorm'] = \
                istop, itn, r1norm, r2norm, anorm, acond, arnorm, xnorm
            self.output = output

            self.x0 = x0
        solve_time = time.time() - solve_time

        self.output['duration'] = solve_time

        if self.A is None:
            self.output['error'] = "Empty objective"
            return self.output

    def analyze(self):
        if 'error' in self.output:
            return
        x_last, error, self.output = LS_postprocess([self.x0], self.x0, self.A,
                                                    self.b, self.x_true,
                                                    output=self.output,
                                                    is_x=True)
        self.output['iters'], self.output['times'] = [0], [0]

if __name__ == "__main__":
    import unittest
    from comparison.tests.test_solver_lsqr import TestSolverLSQR
    unittest.main()
