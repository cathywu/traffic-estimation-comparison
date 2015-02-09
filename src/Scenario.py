import ipdb
import time

from pprint import pprint
import logging
import random

import config as c

# Dependencies for data generation
from synthetic_traffic.networks.EquilibriumNetwork import EquilibriumNetwork
from synthetic_traffic.networks.GridNetwork import GridNetwork
from synthetic_traffic.sensors.SensorConfiguration import SensorConfiguration

# Import solvers
from SolverLS import SolverLS
from SolverBI import SolverBI
from SolverCS import SolverCS
from SolverLSQR import SolverLSQR

# Dependencies for Bayesian inference
from bayesian.grid_model import load_model, create_model
from bayesian.grid_simulation import MCMC

# Dependencies for least squares
from BSC_NNLS.python.util import load_data, solver_input

# Dependencies for compressed sensing

# Dependencies for traffic assignment
import scipy.io
import numpy as np

# from isttt2014_experiments import synthetic_data
# from linkpath import LinkPath
# import path_solver
# import Waypoints as WP

from synthetic_traffic.synth_utils import to_sp, array, deprecated
import synthetic_traffic.networks.grid_networks.static_matrix as static_matrix

from scenario_utils import parser, update_args

class Scenario:
    def __init__(self, TN=None, S=None, solver=None, args=None, myseed=None):
        # Save seed for reproducibility
        if myseed is None:
            myseed = random.randint(0,4294967295)
        np.random.seed(myseed)
        random.seed(myseed)
        self.myseed = myseed

        self.args = args if args is not None else self._new_args()
        self.TN = TN if TN is not None else self._new_traffic_network()
        self.S = S if S is not None else self._new_sensor_configuration()
        self.solver = solver if solver is not None else self._new_solver()

        self.output = None

    def _new_args(self):
        p = parser()
        args = p.parse_args()
        return args

    def _new_traffic_network(self):
        if self.args.model == 'P':
            type = 'small_graph_OD.mat' if self.args.sparse \
                else 'small_graph_OD_dense.mat'

            TN = GridNetwork(nrow=self.args.nrow, ncol=self.args.ncol,
                             nodroutes=self.args.nodroutes, myseed=self.myseed)
            if type == 'small_graph_OD.mat':
                TN.sample_OD_flow(o_flow=1.0,nnz_oroutes=10)
            elif type == 'small_graph_OD_dense.mat':
                TN.sample_OD_flow(o_flow=1.0,sparsity=0.1)
        else:
            SO = True if self.args.model == 'SO' else False

            TN = EquilibriumNetwork(SO=SO,path='los_angeles_data_2.mat')
        return TN

    def _new_sensor_configuration(self):
        return SensorConfiguration(num_link=np.inf, num_OD=np.inf,
                                   num_cellpath_NB=self.args.NB,
                                   num_cellpath_NL=self.args.NL,
                                   num_cellpath_NS=self.args.NS,
                                   num_linkpath=self.args.NLP)

    def _new_solver(self):
        eq = 'CP' if self.args.use_CP else 'OD'
        if self.args.solver == 'CS':
            solver = SolverCS(self.args, full=self.args.all_links,
                                   L=self.args.use_L, OD=self.args.use_OD,
                                   CP=self.args.use_CP, LP=self.args.use_LP, eq=eq)
        elif self.args.solver == 'BI':
            solver = SolverBI(self.args.sparse, full=self.args.all_links,
                                   L=self.args.use_L, OD=self.args.use_OD,
                                   CP=self.args.use_CP, LP=self.args.use_LP)
        elif self.args.solver == 'LS':
            solver = SolverLS(self.args, full=self.args.all_links,
                                   init=self.args.init, L=self.args.use_L,
                                   OD=self.args.use_OD, CP=self.args.use_CP,
                                   LP=self.args.use_LP, eq=eq)
        elif self.args.solver == 'LSQR':
            solver = SolverLSQR(self.args, full=self.args.all_links,
                                     L=self.args.use_L, OD=self.args.use_OD,
                                     CP=self.args.use_CP, LP=self.args.use_LP)
        else:
            return NotImplemented
        return solver

    def run(self):
        # Generate data output
        self.S.sample_sensors(self.TN)
        data = self.S.export_matrices(self.TN)
        if 'error' in data:
            return {'error' : data['error']}

        self.data = data

        self.solver.setup(self.data)
        self.solver.solve()
        self.solver.analyze()

        if self.args.output == True:
            pprint(self.solver.output)

        self.output = self.solver.output

if __name__ == "__main__":
    # output = scenario()
    myseed = 9374293

    # use argparse object as default template
    p = parser()
    args = p.parse_args()
    # if params is not None:
    #     args = update_args(args, params)
    if args.log in c.ACCEPTED_LOG_LEVELS:
        logging.basicConfig(level=eval('logging.'+args.log))

    scen = Scenario(args=args)
    scen.run()

    ## CS experiment
    ## TODO: invoke matlab?
    # experiment_CS(test)

    ## TA experiment
    ## TODO: invoke matlab or cvxopt?
    # delaytype='Polynomial'
    # if delaytype == 'Polynomial': theta = matrix([0.0, 0.0, 0.0, 0.15, 0.0, 0.0])
    # if delaytype == 'Hyperbolic': theta = (3.5, 3.0)
    # g = los_angeles(theta, delaytype, path=path)[3]

    ## Comparison plot




