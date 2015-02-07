import unittest
try:
    from comparison.src.scenario_utils import parser
    from comparison.src.Scenario import Scenario
except ImportError:
    from src.scenario_utils import parser
    from src.Scenario import Scenario

__author__ = 'cathywu'

class TestSolverBI(unittest.TestCase):

    def setUp(self):
        # use argparse object as default template
        p = parser()
        self.args, unknown = p.parse_known_args()
        self.args.solver = 'BI'

    def test_p(self):
        self.args.model = 'P'
        self.args.nrow = 2
        self.args.ncol = 2
        self.args.use_L = False
        scen = Scenario(args=self.args)
        scen.run()
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()