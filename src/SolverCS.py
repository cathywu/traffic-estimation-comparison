from __future__ import division

import ipdb
import time
import sys
import logging
from pprint import pprint

import numpy as np
import scipy

import config as c
from Solver import Solver

from synthetic_traffic.synth_utils import array
from scenario_utils import LS_postprocess, CS_solve
from BSLS.python.bsls_matrices import BSLSMatrices


class SolverCS(Solver):
    def __init__(self,  iters=6000, test='temp', full=True, L=True, OD=True,
                 CP=True, LP=True, eq='CP', init=False, noise=0,
                 method='py_oracle'):
        Solver.__init__(self, test=test, full=full, L=L, OD=OD, CP=CP, LP=LP,
                        eq=eq, init=init, noise=noise, method=method)

        self.iters = iters

        # CS test config
        self.CS_PATH = '/Users/cathywu/Dropbox/Fa13/EE227BT/traffic-project'
        self.OUT_PATH = '%s/' % c.DATA_DIR

        # Test parameters
        # self.method = 'cvx_random_sampling_L1_30_replace'
        # self.method = method  # 'cvx_random_sampling_L1_6000_replace'
        # self.method = 'cvx_oracle'
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
        self.A, self.b, self.N, self.block_sizes, self.x_true, self.nz, \
            self.flow, self.rsort_index, self.x0 = None, None, None, None, \
            None, None, None, None, None
        self.fname, self.mlab = None, None

    def setup(self, data):
        init_time = time.time()
        config = {
            'full': self.full, 'L': self.L, 'OD': self.OD, 'CP': self.CP,
            'LP': self.LP, 'eq': self.eq, 'init': self.init,
        }

        self.fname = '%s/%s' % (c.DATA_DIR, self.test) if data is None else None
        bm = BSLSMatrices(data=data, fname=self.fname, **config)
        bm.degree_reduced_form()
        self.A, self.b, self.N, self.block_sizes, self.x_true, self.nz,\
            self.flow, self.rsort_index, self.x0, self.C = bm.get_CS()
        init_time = time.time() - init_time
        self.output = bm.info
        self.output['init_time'] = init_time

        assert np.linalg.norm(self.C.dot(self.x_true) - \
                              np.ones(self.C.shape[0])) < 1e-10, 'Ux!=1'

        # x0 = np.array(util.block_e(block_sizes - 1, block_sizes))

        if self.noise:
            # b_true = self.b
            delta = np.random.normal(scale=self.b*self.noise)
            self.b = self.b + delta

        if self.block_sizes is not None:
            logging.debug("Blocks: %s" % self.block_sizes.shape)
        # z0 = np.zeros(N.shape[1])

        self.fname = '%s/CS_%s' % (c.DATA_DIR, self.test)
        try:
            assert np.linalg.norm(self.A.dot(self.x_true)-self.b) <= 1e-6, \
                'Ax!=b'
            scipy.io.savemat(self.fname, { 'A': self.A, 'b': self.b,
                                      'x_true': self.x_true, 'flow': self.flow,
                                      'x0': self.x0, 'U': self.C,
                                      'block_sizes': self.block_sizes},
                             oned_as='column')
        except TypeError:
            pprint({ 'A': self.A, 'b': self.b, 'x_true': self.x_true,
                     'flow': self.flow, 'x0': self.x0, 'U': self.C,
                     'block_sizes': self.block_sizes })
            self.output['error'] = 'Problem saving matrices, likely A is empty'
            return

    def solve(self):
        if 'error' in self.output:
            return

        if self.block_sizes is not None and len(self.block_sizes) == \
                self.A.shape[1]:
            self.output['error'] = "Trivial example: nblocks == nroutes"
            logging.error(self.output['error'])
            return

        duration_time = time.time()
        if self.method in ['py_random_sampling_L1_replace','py_oracle']:
            if self.method == 'py_oracle':
                self.x, obj_init, obj_val = self.oracle()
            else:
                self.x, obj_init, obj_val = self.random_sampling(iters=self.iters)
            self.output['f_infty(x)'] = obj_val
            self.output['f_infty(x_init)'] = obj_init
        elif self.method in ['cvx_random_sampling_L1_6000_replace',
                             'cvx_random_sampling_L1_30_replace',
                             'cvx_oracle']:
            self.x = self.random_sampling_matlab()
        duration_time = time.time() - duration_time
        self.output['duration'], self.output['iters'], self.output['times'] = \
            duration_time, [0], [0]

    def analyze(self):
        self.mlab = None
        if 'error' in self.output:
            return
        x0 = self.maximal_support(self.x_true, self.block_sizes)
        x_last, error, self.output = LS_postprocess([self.x], x0, self.A,
                                                    self.b, self.x_true,
                                                    scaling=self.flow,
                                                    block_sizes=self.block_sizes,
                                                    N=self.N, output=self.output,
                                                    is_x=True)

    @staticmethod
    def sample_mask(prior, cum_blocks):
        mask = np.zeros(prior.shape)
        blocks_start = cum_blocks
        blocks_end = cum_blocks[1:]
        for (s,e) in zip(blocks_start,blocks_end):
            try:
                mask[np.random.choice(range(s,e), 1,
                                      p=prior[s:e]/sum(prior[s:e]))] = 1
            except ValueError:
                import ipdb
                ipdb.set_trace()
        assert(np.sum(mask) == cum_blocks.size-1)
        return mask

    def random_sampling(self,iters=6000):
        prior = np.abs(self.x0)
        min_val = np.inf
        min_x = None
        cum_blocks = np.concatenate(([0], np.cumsum(self.block_sizes)))

        obj_val, init_val = np.inf, np.inf
        for iter in xrange(iters):
            print '\rIteration: %d (%0.5f --> %0.5f)' % (iter, init_val, obj_val),
            sys.stdout.flush()
            mask = SolverCS.sample_mask(prior, cum_blocks)
            x_next, init_val, obj_val = CS_solve(self.A, self.b, mask, self.N,
                                   self.block_sizes, mask)
            if obj_val <= min_val:
                min_val = obj_val
                min_x = x_next
                prior = x_next # replace the prior
        return min_x, min_val

    @staticmethod
    def maximal_support(x,block_sizes):
        cum_blocks = np.concatenate(([0], np.cumsum(block_sizes)))
        mask = np.zeros(x.shape)
        blocks_start = cum_blocks
        blocks_end = cum_blocks[1:]
        for (s,e) in zip(blocks_start,blocks_end):
            mask[s + np.argmax(x[s:e])] = 1
        assert(np.sum(mask) == cum_blocks.size-1)
        return mask

    def oracle(self):
        # Try solving the problem without sparsity first
        # iters, times, states = solve_in_z(self.A, self.b, self.x0, self.N,
        #                                   self.block_sizes, 'BB')
        # x1 = self.x0 + self.N.dot(states[-1])
        mask = self.maximal_support(self.x_true, self.block_sizes)
        x, obj_init, obj_val = CS_solve(self.A, self.b, mask, self.N, self.block_sizes,
                              mask)
        return x, obj_init, obj_val

    def random_sampling_matlab(self):
        # Perform test via MATLAB
        from pymatbridge import Matlab
        mlab = Matlab()
        mlab.start()
        mlab.run_code('cvx_solver mosek;')
        mlab.run_code("addpath '~/mosek/7/toolbox/r2012a';")
        self.mlab = mlab

        if self.block_sizes is not None and len(self.block_sizes) == \
                self.A.shape[1]:
            self.output['error'] = "Trivial example: nblocks == nroutes"
            logging.error(self.output['error'])
            self.mlab.stop()
            self.mlab = None
            return

        duration_time = time.time()
        p = self.mlab.run_func('%s/scenario_to_output.m' % self.CS_PATH,
                               { 'filename' : self.fname, 'type': self.test,
                                 'algorithm' : self.method,
                                 'outpath' : self.OUT_PATH })
        duration_time = time.time() - duration_time
        self.mlab.stop()
        return array(p['result'])

if __name__ == "__main__":
    import unittest
    from comparison.tests.test_solver_cs import TestSolverCS
    unittest.main()
