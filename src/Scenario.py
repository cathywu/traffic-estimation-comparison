import config as c
from grid_networks import static_matrix

# Dependencies for Bayesian inference
from grid_model import create_model
from grid_simulation import MCMC

# Dependencies for least squares

# Dependencies for compressed sensing

# Dependencies for traffic assignment
from generate_graph import los_angeles
from cvxopt import matrix
from isttt2014_experiments import synthetic_data
import path_solver
import Waypoints as WP
import scipy.io
import numpy as np
from scipy.sparse import csr_matrix

# Helper functions
# -------------------------------------
def to_np(X):
    return np.array(X).squeeze()

def to_sp(X):
    return csr_matrix((to_np(X.V),(to_np(X.I),to_np(X.J))), shape=X.size)

def generate_data_P(num_rows=2, num_cols=6, num_routes_per_od=4,num_nonzero_routes_per_o=10, prefix=''):
    """
    Generate and export probabilistic matrices

    :param num_rows:
    :param num_cols:
    :param num_routes_per_od:
    :param num_nonzero_routes_per_o:
    :param prefix:
    :return:
    """
    static_matrix.export_matrices(prefix,num_rows,num_cols,num_routes_per_od,
                                  num_nonzero_routes_per_o)

def generate_data_UE(data=None, SO=False, trials=10, demand=3, N=10,
                     plot=False, withODs=False, prefix=''):
    """
    Generate and export UE matrices

    :param data:
    :param SO:
    :param trials:
    :param demand:
    :param N:
    :param plot:
    :param withODs:
    :param prefix:
    :return:
    """
    # FIXME copy mat file to local directory?
    path='/Users/cathywu/Dropbox/PhD/traffic-estimation-wardrop/los_angeles_data_2.mat'
    g, x_true, l, path_wps, wp_trajs, obs = synthetic_data(data, SO, demand, N,
                                                           path=path)
    obs=obs[0]
    A_full = path_solver.linkpath_incidence(g)
    U,f = WP.simplex(g, wp_trajs, withODs)
    T,d = path_solver.simplex(g)
    if not SO:
        fname = 'UE_graph.mat'
    else:
        fname = "SO_graph.mat"
    # Export
    scipy.io.savemat(prefix + fname, {'A_full': to_sp(A_full),
                                      'b_full': to_np(l),
                                      'A': to_sp(A_full[obs,:]),
                                      'b': to_np(l[obs]),
                                      'x_true': to_np(x_true),
                                      'T': to_sp(T),'d': to_np(d),
                                      'U': to_sp(U), 'f': to_np(f)},
                     oned_as='column')

if __name__ == "__main__":
    prefix = c.DATA_DIR + '/'

    # Generate data
    generate_data_P(prefix=prefix)
    print "Generated probabilistic data"
    # TODO: what does this mean?
    # N0, N1, scale, regions, res, margin
    # data = (20, 40, 0.2, [((3.5, 0.5, 6.5, 3.0), 20)], (12,6), 2.0)
    data = (5, 10, 0.2, [((3.5, 0.5, 6.5, 3.0), 5)], (6,3), 2.0)
    # generate_data_UE(data=data, prefix=prefix)
    # generate_data_UE(data=data, SO=True, prefix=prefix)
    # print "Generated equilibrium data"

    sparse = False

    ## LS experiment
    ## TODO: invoke solver

    ## BI experiment
    test = 'UE_graph.mat'
    model = create_model('%s/%s' % (c.DATA_DIR,test),sparse, OD=True, CP=True)

    # test = 'small_graph_OD_dense.mat'
    # model = create_model('%s/%s' % (c.DATA_DIR,test),sparse)
    trace = MCMC(model)

    ## CS experiment
    ## TODO: invoke matlab?

    ## TA experiment
    ## TODO: invoke matlab or cvxopt?
    # delaytype='Polynomial'
    # if delaytype == 'Polynomial': theta = matrix([0.0, 0.0, 0.0, 0.15, 0.0, 0.0])
    # if delaytype == 'Hyperbolic': theta = (3.5, 3.0)
    # g = los_angeles(theta, delaytype, path=path)[3]

    ## Comparison plot




