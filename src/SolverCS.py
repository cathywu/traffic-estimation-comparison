import time
import logging
from pprint import pprint

import numpy as np
import scipy

import config as c
from Solver import Solver

from synthetic_traffic.synth_utils import array
from scenario_utils import LS_postprocess

class SolverCS(Solver):
    def __init__(self, test='temp', full=False, L=True, OD=True, CP=True,
                 LP=True, eq='CP', init=False, noise=0,
                 method='cvx_random_sampling_L1_30_replace'):
        Solver.__init__(self)

        self.test = test
        self.eq = eq
        self.init = init
        self.full = full
        self.L = L
        self.OD = OD
        self.CP = CP
        self.LP = LP
        self.noise = noise

        # CS test config
        self.CS_PATH = '/Users/cathywu/Dropbox/Fa13/EE227BT/traffic-project'
        self.OUT_PATH = '%s/' % c.DATA_DIR

        # Test parameters
        self.method = 'cvx_random_sampling_L1_30_replace'
        # alg = 'cvx_oracle'
        # alg = 'cvx_unconstrained_L1'
        # alg = 'cvx_L2'
        # alg = 'cvx_raw'
        # alg = 'cvx_weighted_L1'
        # alg = 'cvx_hot_start_lp'
        # alg = 'cvx_single_block_L_infty'
        # alg = 'cvx_random_sample_L_infty'
        # alg = 'cvx_mult_blocks_L_infty'
        # alg = 'cvx_block_descent_L_infty'
        # alg = 'cvx_entropy'

    def setup(self, data):
        import config as c
        init_time = time.time()
        if data is None and self.test is not None:
            self.fname = '%s/%s' % (c.DATA_DIR,self.test)
            from python.util import solver_input, load_data
            self.A, self.b, self.N, self.block_sizes, self.x_true, self.nz,\
            self.flow, self.rsort_index, self.x0, out = \
                load_data(self.fname, full=self.full, L=self.L, OD=self.OD,
                          CP=self.CP, LP=self.LP, eq=self.eq, init=self.init)
        else:
            from python.util import solver_input, load_data
            self.A, self.b, self.N, self.block_sizes, self.x_true, self.nz, \
            self.flow, self.rsort_index, self.x0, out = \
                solver_input(data, full=self.full, L=self.L, OD=self.OD,
                          CP=self.CP, LP=self.LP, eq=self.eq, init=self.init)
        init_time = time.time() - init_time
        self.output = out
        self.output['init_time'] = init_time

        # x0 = np.array(util.block_e(block_sizes - 1, block_sizes))

        if self.noise:
            b_true = self.b
            delta = np.random.normal(scale=self.b*self.noise)
            self.b = self.b + delta

        if self.block_sizes is not None:
            logging.debug("Blocks: %s" % self.block_sizes.shape)
        # z0 = np.zeros(N.shape[1])

        self.fname = '%s/CS_%s' % (c.DATA_DIR,self.test)
        try:
            scipy.io.savemat(self.fname, { 'A': self.A, 'b': self.b,
                                      'x_true': self.x_true, 'flow' : self.flow,
                                      'x0': self.x0, 'block_sizes': self.block_sizes},
                             oned_as='column')
        except TypeError:
            pprint({ 'A': self.A, 'b': self.b, 'x_true': self.x_true,
                     'flow' : self.flow, 'x0': self.x0,
                     'block_sizes': self.block_sizes })
            import ipdb
            ipdb.set_trace()

        # Perform test via MATLAB
        from pymatbridge import Matlab
        mlab = Matlab()
        mlab.start()
        mlab.run_code('cvx_solver mosek;')
        mlab.run_code("addpath '~/mosek/7/toolbox/r2012a';")
        self.mlab = mlab

    def solve(self):
        if self.block_sizes is not None and len(self.block_sizes) == self.A.shape[1]:
            self.output['error'] = "Trivial example: nblocks == nroutes"
            logging.error(self.output['error'])
            return

        duration_time = time.time()
        p = self.mlab.run_func('%s/scenario_to_output.m' % self.CS_PATH,
                          { 'filename' : self.fname, 'type' : self.test,
                            'algorithm' : self.method,
                            'outpath' : self.OUT_PATH })
        duration_time = time.time() - duration_time
        self.mlab.stop()
        self.x = array(p['result'])
        self.output['duration'], self.output['iters'], self.output['times'] = \
            duration_time, [0], [0]

    def analyze(self):
        if 'error' in self.output:
            return
        x_last, error, self.output = LS_postprocess([self.x],self.x,self.A,self.b,
                                               self.x_true,scaling=self.flow,
                                               block_sizes=self.block_sizes,
                                               N=self.N,output=self.output,is_x=True)

if __name__ == "__main__":
    import unittest
    from comparison.tests.test_solver_cs import TestSolverCS
    unittest.main()
