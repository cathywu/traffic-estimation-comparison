
import numpy as np

import config as c
from experiment_utils import generate_grid_networks, \
    generate_equilibrium_networks, generate_sensor_configurations

from SolverBI import SolverBI
from SolverLSQR import SolverLSQR
from SolverLS import SolverLS
from SolverCS import SolverCS
from scenario_utils import save

def grid_networks(myseed=None):
    times = 1
    nrows = range(1,5,3)
    ncols = range(2,5,2)
    nodroutes = [15]
    sparsity = [0.1, 0.25, 0.5, 0.75, 1]

    generate_grid_networks(nrows,ncols,nodroutes,times=times,myseed=myseed)

def UE_networks():
    SOs = [True, False]
    EQ_network_path = 'los_angeles_data_2.mat'
    generate_equilibrium_networks(SOs=SOs,path=EQ_network_path)

def sensor_configurations(myseed=None):
    # SENSOR NETWORKS
    num_links = [0, np.inf]
    num_ODs = [0, np.inf]
    num_cellpath_NBs = [5,7,8,9,14,16,29,38,51] # range(0,300,30)
    num_cellpath_NLs = [0] # [0, 100, np.inf]
    num_cellpath_NSs = [0]
    num_linkpaths = [1,2,3,5,6,8] # range(0,300,30)
    myseed = 2347234328
    times = 1

    generate_sensor_configurations(num_links=num_links,num_ODs=num_ODs,
                                   num_cellpath_NBs=num_cellpath_NBs,
                                   num_cellpath_NLs=num_cellpath_NLs,
                                   num_cellpath_NSs=num_cellpath_NSs,
                                   num_linkpaths=num_linkpaths,
                                   times=times,myseed=myseed)

def solvers():
    solvers = []
    # PRIORITY 1
    # ---------------------------------------------------------------
    # solvers.append(SolverLSQR(damp=0))
    # solvers.append(SolverCS(method='cvx_random_sampling_L1_30_replace'))

    # solvers.append(SolverLS(init=True,method='BB'))
    # solvers.append(SolverLS(init=False,method='BB'))

    solvers.append(SolverBI(sparse=True))
    solvers.append(SolverBI(sparse=False))

    # PRIORITY 2
    # ---------------------------------------------------------------
    # solvers.append(SolverBI(sparse=0))
    # solvers.append(SolverBI(sparse=1))
    # solvers.append(SolverBI(sparse=2))

    # solvers.append(SolverLS(init=True,method='LBFGS'))
    # solvers.append(SolverLS(init=False,method='LBFGS'))
    # solvers.append(SolverLS(init=True,method='DORE'))
    # solvers.append(SolverLS(init=False,method='DORE'))

    # solvers.append(SolverCS(method='cvx_oracle'))
    # solvers.append(SolverLSQR(damp=1))

    for s in solvers:
        save(s, prefix="%s/Solver" % c.SOLVER_DIR)

if __name__ == "__main__":
    myseed = 2347234328
    grid_networks(myseed=myseed)
    # UE_networks()
    sensor_configurations(myseed=myseed)
    solvers()



