
class Experiment:

    def __init__(self, tn_dir, sc_dir, solver_dir, scenario_dir):
        self.tn_dir = tn_dir
        self.sc_dir = sc_dir
        self.solver_dir = solver_dir
        self.scenario_dir = scenario_dir

        self.done = {}

    def scan_done(self):
        # Read scenario dir and populate done with combinations
        pass

    def start(self):
        # Kill test after 10 minutes if it's not done yet
        # Parallelize?
        # Randomly select test
        # give/issue some priorities?
        pass