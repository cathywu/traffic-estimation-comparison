import time
import logging

import numpy as np

from Solver import Solver

# FIXME temporary hack
try:
    from scenario_utils import LS_postprocess, LS_solve
except ImportError:
    import config as c

class SolverLS(Solver):
    def __init__(self, args, test=None, data=None, full=True, L=True, OD=True,
                  CP=True, LP=True, eq='CP', init=True):
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

    def setup(self, data):
        init_time = time.time()
        if data is None and self.test is not None:
            from BSC_NNLS.python.util import load_data
            fname = '%s/%s' % (c.DATA_DIR,self.test)
            self.A, self.b, self.N, self.block_sizes, self.x_true, self.nz,\
            self.flow, self.rsort_index, self.x0, out = \
                load_data(fname, full=self.full, L=self.L, OD=self.OD,
                          CP=self.CP, LP=self.LP, eq=self.eq, init=self.init)
        else:
            from BSC_NNLS.python.util import solver_input
            self.A, self.b, self.N, self.block_sizes, self.x_true, self.nz, \
            self.flow, self.rsort_index, self.x0, out = \
                solver_input(data, full=self.full, L=self.L, OD=self.OD,
                          CP=self.CP, LP=self.LP, eq=self.eq, init=self.init)
        init_time = time.time() - init_time
        self.output = out
        self.output['init_time'] = init_time

        # x0 = np.array(util.block_e(block_sizes - 1, block_sizes))

        if self.args.noise:
            b_true = self.b
            delta = np.random.normal(scale=self.b*self.args.noise)
            self.b = self.b + delta

        if self.block_sizes is not None:
            logging.debug("Blocks: %s" % self.block_sizes.shape)
        # z0 = np.zeros(N.shape[1])

    def solve(self):
        if self.N is None or (self.block_sizes-1).any() == False:
            self.iters, self.times, self.states = [0],[0],[self.x0]
        else:
            self.iters, self.times, self.states = LS_solve(self.A,self.b,self.x0,self.N,
                                            self.block_sizes,self.args)

    def analyze(self):
        x_last, error, self.output = LS_postprocess(self.states,self.x0,self.A,self.b,
                                               self.x_true,scaling=self.flow,
                                               block_sizes=self.block_sizes,N=self.N,
                                               output=self.output)

if __name__ == "__main__":
    import unittest
    import config
    import ipdb
    from comparison.tests.test_solver_ls import TestSolverLS
    unittest.main()
