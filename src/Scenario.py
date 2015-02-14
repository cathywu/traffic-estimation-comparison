import ipdb
import logging
import random
from pprint import pprint

import numpy as np

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

from scenario_utils import parser, update_args, save

class Scenario:
    def __init__(self, TN=None, SC=None, solver=None, args=None, myseed=None,
                 fname_tn=None, fname_sc=None, fname_solver=None, test=False):
        # Save seed for reproducibility
        if myseed is None:
            myseed = random.randint(0,4294967295)
        np.random.seed(myseed)
        random.seed(myseed)
        self.myseed = myseed
        self.test = test

        # Need to set this before the next few steps
        self.args = args if args is not None else self._new_args()

        self._init_traffic_network(TN,fname_tn)
        self._init_sensor_configuration(SC,fname_sc)
        self._init_solver(solver,fname_solver)

        self.output = None

    def _init_traffic_network(self,TN,fname_tn):
        if fname_tn is not None:
            self.fname_tn = fname_tn
            import pickle
            if self.test:
                fpath = '%s/test/%s' % (c.TN_DIR,fname_tn)
            else:
                fpath = '%s/%s' % (c.TN_DIR,fname_tn)
            with open(fpath) as f:
                self.TN = pickle.load(f)
        else:
            self.TN = TN if TN is not None else self._new_traffic_network()

    def _init_sensor_configuration(self,SC,fname_sc):
        if fname_sc is not None:
            self.fname_sc = fname_sc
            import pickle
            if self.test:
                fpath = '%s/test/%s' % (c.SC_DIR,fname_sc)
            else:
                fpath = '%s/%s' % (c.SC_DIR,fname_sc)
            with open(fpath) as f:
                self.SC = pickle.load(f)
        else:
            self.SC = SC if SC is not None else self._new_sensor_configuration()

    def _init_solver(self,solver,fname_solver):
        if fname_solver is not None:
            self.fname_solver = fname_solver
            import pickle
            if self.test:
                fpath = '%s/test/%s' % (c.SOLVER_DIR,fname_solver)
            else:
                fpath = '%s/%s' % (c.SOLVER_DIR,fname_solver)
            with open(fpath) as f:
                self.solver = pickle.load(f)
        else:
            self.solver = solver if solver is not None else self._new_solver()

    def save(self, prefix='%s/Scenario'):
        print "SAVING EXPERIMENT"
        save(self, prefix=prefix % c.SCENARIO_DIR_NEW)

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
            solver = SolverCS(full=self.args.all_links, L=self.args.use_L,
                              OD=self.args.use_OD, CP=self.args.use_CP,
                              LP=self.args.use_LP, eq=eq)
        elif self.args.solver == 'BI':
            solver = SolverBI(sparse=self.args.sparse, full=self.args.all_links,
                              L=self.args.use_L, OD=self.args.use_OD,
                              CP=self.args.use_CP, LP=self.args.use_LP)
        elif self.args.solver == 'LS':
            solver = SolverLS(full=self.args.all_links, init=self.args.init,
                              L=self.args.use_L, OD=self.args.use_OD,
                              CP=self.args.use_CP, LP=self.args.use_LP, eq=eq,
                              noise=self.args.noise, method=self.args.method)
        elif self.args.solver == 'LSQR':
            solver = SolverLSQR(full=self.args.all_links, L=self.args.use_L,
                                OD=self.args.use_OD, CP=self.args.use_CP,
                                LP=self.args.use_LP)
        else:
            return NotImplemented
        return solver

    def run(self):
        # Generate data output
        self.SC.sample_sensors(self.TN)
        data = self.SC.export_matrices(self.TN)
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





