
import numpy as np

import config as c
from experiment_utils import generate_grid_networks, \
    generate_equilibrium_networks, generate_sensor_configurations

from SolverBI import SolverBI
from SolverLSQR import SolverLSQR
from SolverLS import SolverLS
from SolverCS import SolverCS
from scenario_utils import save


def grid_networks_small(myseed=None):
    times = 1
    nrows = range(1, 5, 3)
    ncols = range(2, 5, 2)
    nodroutes = [15]

    generate_grid_networks(nrows, ncols, nodroutes, times=times, myseed=myseed)


def grid_networks_small_enough(myseed=None):
    times = 1
    nrows = range(1, 11, 3)
    ncols = range(2, 11, 2)
    nodroutes = [15]

    generate_grid_networks(nrows, ncols, nodroutes, times=times, myseed=myseed,
                           max_prod=40)

def grid_networks_all(myseed=None):
    times = 1
    nrows = range(1, 11, 3)
    ncols = range(2, 11, 2)
    nodroutes = [15]

    generate_grid_networks(nrows, ncols, nodroutes, times=times, myseed=myseed)

def UE_networks():
    SOs = [True, False]
    EQ_network_path = 'los_angeles_data_2.mat'
    generate_equilibrium_networks(SOs=SOs, path=EQ_network_path)


def sensor_configurations_small_enough(myseed=None):
    # SENSOR NETWORKS
    num_links = [0, np.inf]
    num_ODs = [0, np.inf]
    num_cellpath_NBs = [0, 5, 6, 7, 8, 9, 14, 16, 17, 19, 23, 24, 25, 28, 29,
                        32, 35, 38, 39, 42, 47, 50, 51, 55, 57, 58, 59, 63,
                        68, 69, 70, 74, 77, 590, 79, 89, 97, 98, 99, 101, 102,
                        103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113,
                        114, 115, 116, 117, 118, 119, 120, 123, 124, 125, 126,
                        128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138,
                        139, 142, 144, 145, 146, 147, 148, 150, 151, 155, 156,
                        157, 158, 159, 163, 164, 166, 167, 168, 169, 170, 171,
                        172, 174, 176, 177, 179, 181, 185, 187, 189, 190, 191,
                        193, 194, 196, 197, 198, 199, 201, 202, 203, 204, 206,
                        209, 215, 224, 226, 228, 229, 231, 235, 237, 242, 245,
                        246, 253, 254, 259, 270, 271, 277, 278, 281, 293, 294,
                        295, 296, 300, 304, 324, 326, 331, 332, 342, 345, 359,
                        369, 373, 747, 390, 392, 406, 424, 442, 490, 491]
    # Formerly range(0,300,30)
    num_cellpath_NLs = [0]  # [0, 100, np.inf]
    num_cellpath_NSs = [0]
    num_linkpaths = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                     17, 18, 19, 21, 22, 23, 24, 25, 26, 30, 31, 33, 35, 36,
                     39, 41, 45, 47, 49, 50, 52, 54, 60, 62, 71, 90, 119]
    myseed = 2347234328
    times = 1

    generate_sensor_configurations(num_links=num_links,num_ODs=num_ODs,
                                   num_cellpath_NBs=num_cellpath_NBs,
                                   num_cellpath_NLs=num_cellpath_NLs,
                                   num_cellpath_NSs=num_cellpath_NSs,
                                   num_linkpaths=num_linkpaths,
                                   times=times,myseed=myseed)


def sensor_configurations_small(myseed=None):
    # SENSOR NETWORKS
    num_links = [0, np.inf]
    num_ODs = [0, np.inf]
    num_cellpath_NBs = [5, 7, 8, 9, 14, 16, 29, 38, 51]  # range(0,300,30)
    num_cellpath_NLs = [0]  # [0, 100, np.inf]
    num_cellpath_NSs = [0]
    num_linkpaths = [1, 2, 3, 5, 6, 8]  # range(0,300,30)
    myseed = 2347234328
    times = 1

    generate_sensor_configurations(num_links=num_links, num_ODs=num_ODs,
                                   num_cellpath_NBs=num_cellpath_NBs,
                                   num_cellpath_NLs=num_cellpath_NLs,
                                   num_cellpath_NSs=num_cellpath_NSs,
                                   num_linkpaths=num_linkpaths,
                                   times=times, myseed=myseed)


def solvers(BI=False, LS=False, CS=False, LSQR=False, LS2=False, CS2=False,
            LSQR2=False):
    solvers = []
    # PRIORITY 1
    # ---------------------------------------------------------------
    if LSQR:
        solvers.append(SolverLSQR(damp=0))
    if CS:
        solvers.append(SolverCS(method='cvx_random_sampling_L1_30_replace'))

    if LS:
        solvers.append(SolverLS(init=True, method='BB'))
        solvers.append(SolverLS(init=False, method='BB'))

    if BI:
        solvers.append(SolverBI(sparse=True))
        solvers.append(SolverBI(sparse=False))

    # PRIORITY 2
    # ---------------------------------------------------------------
    if LS2:
        solvers.append(SolverLS(init=True, method='LBFGS'))
        solvers.append(SolverLS(init=False, method='LBFGS'))
        solvers.append(SolverLS(init=True, method='DORE'))
        solvers.append(SolverLS(init=False, method='DORE'))

    if CS2:
        solvers.append(SolverCS(method='cvx_oracle'))

    if LSQR2:
        solvers.append(SolverLSQR(damp=1))

    for s in solvers:
        save(s, prefix="%s/Solver" % c.SOLVER_DIR)


def experiment(BI=False, LS=False, CS=False, LSQR=False, LS2=False, CS2=False,
               LSQR2=False):
    myseed = 2347234328
    # grid_networks_all(myseed=myseed)
    # grid_networks_small_enough(myseed=myseed)
    # UE_networks()
    # sensor_configurations_small_enough(myseed=myseed)
    solvers(BI=BI, LS=LS, CS=CS, LSQR=LSQR, LS2=LS2, CS2=CS2, LSQR2=LSQR2)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--solver', dest='solver', type=str, default='LS',
                        help='Solver to seed experiment')
    args = parser.parse_args()
    if args.solver == 'LS':
        experiment(LS=True)
    elif args.solver == 'BI':
        experiment(BI=True)
    elif args.solver == 'CS':
        experiment(CS=True)
    elif args.solver == 'LSQR':
        experiment(LSQR=True)
    elif args.solver == 'LS2':
        experiment(LS2=True)
    elif args.solver == 'CS2':
        experiment(CS2=True)
    elif args.solver == 'LSQR2':
        experiment(LSQR2=True)
