import unittest
try:
    from comparison.src.scenario_utils import parser
    from comparison.src.Scenario import Scenario
except ImportError:
    from src.scenario_utils import parser
    from src.Scenario import Scenario

__author__ = 'cathywu'

class TestSolverLS(unittest.TestCase):

    def setUp(self):
        # use argparse object as default template
        p = parser()
        self.args, unknown = p.parse_known_args()
        self.args.solver = 'LS'

    def test_p0(self):
        self.args.model = 'P'
        self.args.nrow = 5
        self.args.ncol = 2
        scen = Scenario(args=self.args)
        scen.run()
        self.assertTrue(True)

    def test_p1(self):
        self.args.model = 'P'
        self.args.nrow = 5
        self.args.ncol = 2
        self.args.use_L = False
        scen = Scenario(args=self.args)
        scen.run()
        self.assertTrue(True)

    def test_p2(self):
        self.args.model = 'P'
        self.args.nrow = 5
        self.args.ncol = 2
        self.sparse = True
        scen = Scenario(args=self.args)
        scen.run()
        self.assertTrue(True)

    def test_p3(self):
        self.args.model = 'P'
        self.args.nrow = 5
        self.args.ncol = 2
        self.args.use_L = False
        self.sparse = True
        scen = Scenario(args=self.args)
        scen.run()
        self.assertTrue(True)

    def test_p4(self):
        self.args.model = 'P'
        self.args.nrow = 5
        self.args.ncol = 2
        self.args.use_L = False
        self.args.use_OD = False
        self.sparse = True
        scen = Scenario(args=self.args)
        scen.run()
        self.assertTrue(True)

    def test_p5(self):
        # No equality constraint
        self.args.model = 'P'
        self.args.nrow = 5
        self.args.ncol = 2
        self.args.use_L = False
        self.args.use_OD = False
        self.args.use_CP = False
        self.sparse = True
        scen = Scenario(args=self.args)
        scen.run()
        self.assertTrue(True)

    def test_p6(self):
        self.args.model = 'P'
        self.args.nrow = 5
        self.args.ncol = 2
        self.args.use_L = False
        self.args.use_OD = False
        self.args.use_CP = False
        self.args.use_LP = False
        self.sparse = True
        scen = Scenario(args=self.args)
        scen.run()
        self.assertTrue(True)

    def test_ue(self):
        self.args.model = 'UE'
        scen = Scenario(args=self.args)
        scen.run()
        self.assertTrue(True)

    def test_so(self):
        self.args.model = 'SO'
        scen = Scenario(args=self.args)
        scen.run()
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()