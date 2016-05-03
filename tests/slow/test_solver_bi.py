import unittest
import time

try:
    from comparison.src.scenario_utils import new_namespace
    from comparison.src.Scenario import Scenario
except ImportError:
    from src.scenario_utils import new_namespace
    from src.Scenario import Scenario

__author__ = 'cathywu'

class TestSolverBI(unittest.TestCase):

    def setUp(self):
        # use argparse object as default template
        self.args = new_namespace()
        self.args.solver = 'BI'
        self.starttime = time.time()

    def tearDown(self):
        t = time.time() - self.starttime
        print "%s: %0.3f" % (self.id(), t)

    def test_p(self):
        self.args.model = 'P'
        self.args.nrow = 1
        self.args.ncol = 2
        self.args.use_L = False
        scen = Scenario(args=self.args)
        scen.run()
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()