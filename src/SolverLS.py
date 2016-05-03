import time
import logging

import numpy as np

from Solver import Solver

# FIXME temporary hack
try:
    from scenario_utils import LS_postprocess, solve_in_z
except ImportError:
    import config as c
from BSLS.python.bsls_matrices import BSLSMatrices

class SolverLS(Solver):
    def __init__(self, test=None, data=None, full=True, L=True, OD=True,
                  CP=True, LP=True, eq='CP', init=True, noise=0.0, method='BB'):
        Solver.__init__(self, test=test, full=full, L=L, OD=OD, CP=CP, LP=LP,
                        eq=eq, init=init, noise=noise, method=method)

    def setup(self, data):
        init_time = time.time()
        config = {
            'full': self.full, 'L': self.L, 'OD': self.OD, 'CP': self.CP,
            'LP': self.LP, 'eq': self.eq, 'init': self.init,
            }
        self.fname = '%s/%s' % (c.DATA_DIR, self.test) if data is None else None
        bm = BSLSMatrices(data=data, fname=self.fname, **config)
        bm.degree_reduced_form()
        self.A, self.b, self.N, self.block_sizes, self.x_true, self.nz, \
        self.flow, self.rsort_index, self.x0 = bm.get_LS()
        init_time = time.time() - init_time
        self.output = bm.info
        self.output['init_time'] = init_time

        # x0 = np.array(util.block_e(block_sizes - 1, block_sizes))

        if self.noise:
            b_true = self.b
            delta = np.random.normal(scale=self.b*self.noise)
            self.b = self.b + delta

        if self.block_sizes is not None:
            logging.debug("Blocks: %s" % self.block_sizes.shape)
        # z0 = np.zeros(N.shape[1])

    def solve(self):
        if self.block_sizes is not None and len(self.block_sizes) == self.A.shape[1]:
            self.output['error'] = "Trivial example: nblocks == nroutes"
            logging.error(self.output['error'])
            return

        if self.N is None or (self.block_sizes-1).any() == False:
            iters, times, self.states = [0],[0],[self.x0]
        else:
            iters, times, self.states = solve_in_z(self.A,self.b,self.x0,self.N,
                                            self.block_sizes,self.method)

        self.output['duration'] = np.sum(times)
        self.output['iters'], self.output['times'] = list(iters), list(times)

    def analyze(self):
        if 'error' in self.output:
            return
        x_last, error, self.output = LS_postprocess(self.states,self.x0,self.A,self.b,
                                               self.x_true,scaling=self.flow,
                                               block_sizes=self.block_sizes,N=self.N,
                                               output=self.output)

if __name__ == "__main__":
    import unittest
    import config
    from comparison.tests.test_solver_ls import TestSolverLS
    unittest.main()
