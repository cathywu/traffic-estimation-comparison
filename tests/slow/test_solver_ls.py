import unittest
import time

try:
    from comparison.src.scenario_utils import new_namespace
    from comparison.src.Scenario import Scenario
except ImportError:
    from src.scenario_utils import new_namespace
    from src.Scenario import Scenario

__author__ = 'cathywu'

class TestSolverLS(unittest.TestCase):

    def setUp(self):
        # use argparse object as default template
        self.args = new_namespace()
        self.args.solver = 'LS'
        self.args.nrow = 4
        self.args.ncol = 2
        self.args.NB = 10
        self.starttime = time.time()

    def tearDown(self):
        t = time.time() - self.starttime
        print "%s: %0.3f" % (self.id(), t)

    def test_p0(self):
        self.args.model = 'P'
        scen = Scenario(args=self.args)
        scen.run()
        self.assertTrue(scen.output['0.5norm(Ax-b)^2'][-1] < 1e-16)

    def test_p1(self):
        self.args.model = 'P'
        self.args.use_L = False
        scen = Scenario(args=self.args)
        scen.run()
        self.assertTrue(scen.output['0.5norm(Ax-b)^2'][-1] < 1e-16)

    def test_p2(self):
        self.args.model = 'P'
        self.sparse = True
        scen = Scenario(args=self.args)
        scen.run()
        self.assertTrue(scen.output['0.5norm(Ax-b)^2'][-1] < 1e-16)

    def test_p3(self):
        self.args.model = 'P'
        self.args.use_L = False
        self.sparse = True
        scen = Scenario(args=self.args)
        scen.run()
        self.assertTrue(scen.output['0.5norm(Ax-b)^2'][-1] < 1e-16)

    def test_p4(self):
        self.args.model = 'P'
        self.args.use_L = False
        self.args.use_OD = False
        self.sparse = True
        scen = Scenario(args=self.args)
        scen.run()
        self.assertTrue(scen.output['0.5norm(Ax-b)^2'][-1] < 1e-16)

    def test_p5(self):
        # No equality constraint
        self.args.model = 'P'
        self.args.use_L = False
        self.args.use_OD = False
        self.args.use_CP = False
        self.args.use_LP = False
        self.sparse = True
        scen = Scenario(args=self.args)
        scen.run()
        self.assertTrue('error' in scen.output)

    def test_ue(self):
        self.args.model = 'UE'
        scen = Scenario(args=self.args)
        scen.run()
        self.assertTrue(scen.output['0.5norm(Ax-b)^2'][-1] < 1e-16)

    def test_so(self):
        self.args.model = 'SO'
        scen = Scenario(args=self.args)
        scen.run()
        self.assertTrue(scen.output['0.5norm(Ax-b)^2'][-1] < 1e-16)

if __name__ == '__main__':
    unittest.main()