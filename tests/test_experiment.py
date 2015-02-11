import unittest

import numpy as np

import src.config as c
from src.scenario_utils import save
from src.SolverLSQR import SolverLSQR
from src.generate_sensor_configuration import generate_sensor_configurations
from src.generate_traffic_networks import generate_grid_networks
from src.Experiment import Experiment

__author__ = 'cathywu'

class TestExperiment(unittest.TestCase):

    def setUp(self):
        self._generate_traffic_networks()
        self._generate_sensor_configurations()
        self._generate_solvers()

    def _generate_solvers(self):
        save(SolverLSQR(damp=0), prefix='%s/test/Solver' % c.SOLVER_DIR)

    def _generate_sensor_configurations(self):
        num_links = [0]
        num_ODs = [np.inf]
        num_cellpath_NBs = range(0,300,30)
        num_cellpath_NLs = [100]
        num_cellpath_NSs = [0]
        num_linkpaths = [0,5]
        myseed = 2347234328
        times = 1

        generate_sensor_configurations(num_links=num_links,num_ODs=num_ODs,
                                       num_cellpath_NBs=num_cellpath_NBs,
                                       num_cellpath_NLs=num_cellpath_NLs,
                                       num_cellpath_NSs=num_cellpath_NSs,
                                       num_linkpaths=num_linkpaths,
                                       times=times, myseed=myseed,
                                       prefix='%s/test/SC')

    def _generate_traffic_networks(self):
        myseed = 2347234328
        times = 1

        nrows = [1,3]
        ncols = [2,4]
        nodroutes = [15]

        generate_grid_networks(nrows,ncols,nodroutes,times=times,myseed=myseed,
                               prefix='%s/test/TN_Grid')

    def test_experiment(self):
        tn_dir = '%s/test' % c.TN_DIR
        sc_dir = '%s/test' % c.SC_DIR
        solver_dir = '%s/test' % c.SOLVER_DIR
        scenario_dir = '%s/test' % c.SCENARIO_DIR_NEW
        scan_interval = 3
        sample_attempts = 5
        job_timeout = 30
        njobs = 10

        e = Experiment(tn_dir,sc_dir,solver_dir,scenario_dir,
                       scan_interval=scan_interval,
                       sample_attempts=sample_attempts,job_timeout=job_timeout,
                       test=True)

        e.run_experiment(njobs)

if __name__ == '__main__':
    unittest.main()

    # ClEANUP COMMAND
    # rm data/sensor_configurations/test/* data/networks/test/* data/scenarios/test/* data/solvers/test/*

