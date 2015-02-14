import numpy as np
import logging

import config as c
from Solver import Solver
from BSC_NNLS.python.util import solver_input
from bayesian.grid_model import create_model
from bayesian.grid_simulation import MCMC

# FIXME temporary hack
from scenario_utils import LS_postprocess

class SolverBI(Solver):
    def __init__(self, sparse=False, full=False, L=True, OD=True, CP=True, LP=True):
        Solver.__init__(self)

        self.sparse = sparse
        self.full = full
        self.L = L
        self.OD = OD
        self.CP = CP
        self.LP = LP

    def setup(self, data):
        self.AA, self.bb_obs, self.EQ, self.x_true, self.scaling,\
            self.block_sizes, out = \
            solver_input(data, full=self.full, L=self.L, OD=self.OD,
                         CP=self.CP, LP=self.LP, eq='CP', EQ_elim=False)
        self.output = out

    def solve(self):
        if self.EQ is None:
            self.output['error'] = "EQ constraint matrix is empty"
            logging.error(self.output['error'])
            return
        if self.block_sizes is not None and len(self.block_sizes) == self.AA.shape[1]:
            self.output['error'] = "Trivial example: nblocks == nroutes"
            logging.error(self.output['error'])
            return

        logging.info('A: %s' % repr(self.AA.shape))
        self.model,alpha,self.x_pri = create_model(self.AA, self.bb_obs, self.EQ,
                                         self.x_true, sparse=self.sparse)
        self.output['alpha'] = alpha

    def analyze(self):
        if 'error' in self.output:
            return

        # model = create_model('%s/%s' % (c.DATA_DIR,test),sparse)
        if np.all(self.x_pri==1):
            x_last, error, self.output = LS_postprocess([self.x_pri], self.x_pri,
                                                   self.AA.todense(), self.bb_obs,
                                                   self.x_true, output=self.output,
                                                   is_x=True)
        else:
            model, trace, init_time, duration = MCMC(self.model)
            self.output['init_time'], self.output['duration'] = init_time, duration

            x_blocks = None
            for varname in sorted(trace.varnames):
                # flatten the trace and normalize
                if trace.get_values(varname).shape[1] == 0:
                    continue
                x_block = np.array([x/sum(x) for x in trace.get_values(varname)])
                if x_blocks is not None:
                    x_blocks = np.hstack((x_blocks, x_block))
                else:
                    x_blocks = x_block

            x_last, error, self.output = LS_postprocess(x_blocks, x_blocks[0,:],
                                                   self.AA.todense(),
                                               self.bb_obs, self.x_true,
                                               output=self.output, is_x=True)
        self.output['blocks'] = self.EQ.shape[0] if self.EQ is not None else None

if __name__ == "__main__":
    import unittest
    from comparison.tests.test_solver_bi import TestSolverBI
    unittest.main()
