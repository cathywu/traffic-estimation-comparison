import unittest
import random
import time

import numpy as np

try:
    from comparison.src.scenario_utils import new_namespace
    from comparison.src.Scenario import Scenario
except ImportError:
    from src.scenario_utils import new_namespace
    from src.Scenario import Scenario
from src.SolverCS import SolverCS

__author__ = 'cathywu'

class TestSolverCS(unittest.TestCase):

    def setUp(self):
        seed = 237423433
        np.random.seed(seed)
        random.seed(seed)

        # use argparse object as default template
        self.args = new_namespace()
        self.args.solver = 'CS'
        self.args.nrow = 3
        self.args.ncol = 2
        self.args.use_L = False
        self.args.sparse = True
        self.args.NB = 3
        self.starttime = time.time()

    def tearDown(self):
        t = time.time() - self.starttime
        print "%s: %0.3f" % (self.id(), t)

    def test_sample_mask(self):
        prior = np.array([1, 2, 4, 6, 8, 1, 4, 7])
        cum_blocks = np.array([0, 3, 5, 6, 8])
        mask = SolverCS.sample_mask(prior, cum_blocks)
        ans = np.array([0, 1, 0, 1, 0, 1, 1, 0])
        self.assertTrue(np.all(mask == ans))

    def test_maximal_support(self):
        x = np.array([0, 0, 1, 2, 3, 5, 2, 4, 8, 1, -1, 3, 0])
        block_sizes = np.array([3, 4, 3, 3])
        mask = SolverCS.maximal_support(x, block_sizes)
        ans = np.array([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0])
        self.assertTrue(np.all(mask == ans))

    def test_py_oracle_init(self):
        self.args.model = 'P'
        self.args.init = True
        self.args.method = 'py_oracle'
        scen = Scenario(args=self.args)
        scen.run()
        self.assertTrue(scen.output['f_infty(x)'] < \
                        scen.output['f_infty(x_init)'])

    def test_py_oracle(self):
        self.args.model = 'P'
        self.args.init = False
        self.args.method = 'py_oracle'
        scen = Scenario(args=self.args)
        scen.run()
        import ipdb
        ipdb.set_trace()
        self.assertTrue(scen.output['f_infty(x)'] < \
                        scen.output['f_infty(x_init)'])

    # def test_py_random_sampling(self):
    #     self.args.model = 'P'
    #     self.args.init = False
    #     self.args.method = 'py_random_sampling_L1_replace'
    #     scen = Scenario(args=self.args)
    #     scen.run()
    #     self.assertTrue(scen.output['0.5norm(Ax-b)^2'][-1] < \
    #                     scen.output['0.5norm(Ax_init-b)^2'])

    # def test_ue(self):
    #     self.args.model = 'UE'
    #     scen = Scenario(args=self.args)
    #     scen.run()
    #     self.assertTrue(True)

    # def test_so(self):
    #     self.args.model = 'SO'
    #     scen = Scenario(args=self.args)
    #     scen.run()
    #     self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()