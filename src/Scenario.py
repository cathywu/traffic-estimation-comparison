import ipdb

import argparse
import logging

import config as c

# Dependencies for data generation
from grid_networks import static_matrix

# Dependencies for Bayesian inference
from grid_model import create_model
from grid_simulation import MCMC

# Dependencies for least squares
from python.util import load_data
from python import util
from python.c_extensions.simplex_projection import simplex_projection
from python import BB, LBFGS, DORE, solvers
import numpy.linalg as la
import matplotlib.pyplot as plt


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

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', help='Data file (*.mat)',
                        default='route_assignment_matrices_ntt.mat')
    parser.add_argument('--log', dest='log', nargs='?', const='INFO',
                        default='WARN', help='Set log level (default: WARN)')
    parser.add_argument('--solver',dest='solver',type=str,default='BB',
                        help='Solver name')
    parser.add_argument('--noise',dest='noise',type=float,default=None,
                        help='Noise level')
    return parser

# Data generation
# -------------------------------------
def generate_data_P(num_rows=2, num_cols=6, num_routes_per_od=4,
                    num_nonzero_routes_per_o=10, prefix=''):
    """
    Generate and export probabilistic matrices
    """
    static_matrix.export_matrices(prefix,num_rows,num_cols,num_routes_per_od,
                                  num_nonzero_routes_per_o)

def generate_data_UE(data=None, SO=False, trials=10, demand=3, N=10,
                     plot=False, withODs=False, prefix=''):
    """
    Generate and export UE matrices
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

# Experimentation helper functions
# -------------------------------------
def experiment_BI(test,sparse):
    """
    Bayesian inference experiment
    """
    model = create_model('%s/%s' % (c.DATA_DIR,test),sparse, OD=True, CP=True)

    # model = create_model('%s/%s' % (c.DATA_DIR,test),sparse)

    trace = MCMC(model)

def experiment_TA():
    pass

def experiment_CS():
    pass

def experiment_LS(test):
    """
    Least squares experiment
    :param test:
    :return:
    """
    ## LS experiment
    ## TODO: invoke solver
    A, b, N, block_sizes, x_true, nz, flow, rsort_index = \
        load_data('%s/%s' % (c.DATA_DIR,test), full=False, OD=True, CP=False,
                  eq='OD')

    if args.noise:
        b_true = b
        delta = np.random.normal(scale=b*args.noise)
        b = b + delta

    logging.debug("Blocks: %s" % block_sizes.shape)
    x0 = np.array(util.block_e(block_sizes - 1, block_sizes))
    target = A.dot(x0)-b

    options = { 'max_iter': 5000,
                'verbose': 1,
                'suff_dec': 0.003, # FIXME unused
                'corrections': 500 } # FIXME unused
    AT = A.T.tocsr()
    NT = N.T.tocsr()

    f = lambda z: 0.5 * la.norm(A.dot(N.dot(z)) + target)**2
    nabla_f = lambda z: NT.dot(AT.dot(A.dot(N.dot(z)) + target))

    def proj(x):
        projected_value = simplex_projection(block_sizes - 1,x)
        # projected_value = pysimplex_projection(block_sizes - 1,x)
        return projected_value

    z0 = np.zeros(N.shape[1])

    import time
    iters, times, states = [], [], []
    def log(iter_,state,duration):
        iters.append(iter_)
        times.append(duration)
        states.append(state)
        start = time.time()
        return start

    logging.debug('Starting %s solver...' % args.solver)
    if args.solver == 'LBFGS':
        LBFGS.solve(z0+1, f, nabla_f, solvers.stopping, log=log,proj=proj,
                    options=options)
        logging.debug("Took %s time" % str(np.sum(times)))
    elif args.solver == 'BB':
        BB.solve(z0,f,nabla_f,solvers.stopping,log=log,proj=proj,
                 options=options)
    elif args.solver == 'DORE':
        # setup for DORE
        alpha = 0.99
        lsv = util.lsv_operator(A, N)
        logging.info("Largest singular value: %s" % lsv)
        A_dore = A*alpha/lsv
        target_dore = target*alpha/lsv

        DORE.solve(z0, lambda z: A_dore.dot(N.dot(z)),
                   lambda b: N.T.dot(A_dore.T.dot(b)), target_dore, proj=proj,
                   log=log,options=options)
        A_dore = None
    logging.debug('Stopping %s solver...' % args.solver)

    # Plot some stuff
    d = len(states)
    x_hat = N.dot(np.array(states).T) + np.tile(x0,(d,1)).T
    x_last = x_hat[:,-1]

    logging.debug("Shape of x0: %s" % repr(x0.shape))
    logging.debug("Shape of x_hat: %s" % repr(x_hat.shape))

    starting_error = 0.5 * la.norm(A.dot(x0)-b)**2
    opt_error = 0.5 * la.norm(A.dot(x_true)-b)**2
    diff = A.dot(x_hat) - np.tile(b,(d,1)).T
    error = 0.5 * np.diag(diff.T.dot(diff))

    dist_from_true = np.max(np.abs(x_last-x_true))
    start_dist_from_true = np.max(np.abs(x_last-x0))

    x_diff = x_true - x_last
    print 'incorrect x entries: %s' % x_diff[np.abs(x_diff) > 1e-3].shape[0]
    per_flow = np.sum(np.abs(flow * (x_last-x_true))) / np.sum(flow * x_true)
    print 'percent flow allocated incorrectly: %f' % per_flow
    print '0.5norm(Ax-b)^2: %8.5e\n0.5norm(Ax_init-b)^2: %8.5e\n0.5norm(Ax-b)^2: %8.5e\nmax|x-x_true|: %.2f\nmax|x_init-x_true|: %.2f\n\n\n' % \
          (error[-1], starting_error, opt_error, dist_from_true,
           start_dist_from_true)

    plt.figure()
    plt.hist(x_last)

    plt.figure()
    plt.loglog(np.cumsum(times),error)
    plt.show()

    return iters, times, states

if __name__ == "__main__":
    p = parser()
    args = p.parse_args()
    if args.log in c.ACCEPTED_LOG_LEVELS:
        logging.basicConfig(level=eval('logging.'+args.log))

    prefix = c.DATA_DIR + '/'

    # Generate data
    # generate_data_P(prefix=prefix)
    print "Generated probabilistic data"
    # TODO: what does this mean?
    # N0, N1, scale, regions, res, margin
    # data = (20, 40, 0.2, [((3.5, 0.5, 6.5, 3.0), 20)], (12,6), 2.0)
    data = (5, 10, 0.2, [((3.5, 0.5, 6.5, 3.0), 5)], (6,3), 2.0)
    # generate_data_UE(data=data, prefix=prefix)
    # generate_data_UE(data=data, SO=True, prefix=prefix)
    # print "Generated equilibrium data"

    sparse = False
    # test = 'UE_graph.mat'
    test = 'small_graph_OD_dense.mat'
    # experiment_BI(test,sparse)
    experiment_LS(test)

    ## CS experiment
    ## TODO: invoke matlab?

    ## TA experiment
    ## TODO: invoke matlab or cvxopt?
    # delaytype='Polynomial'
    # if delaytype == 'Polynomial': theta = matrix([0.0, 0.0, 0.0, 0.15, 0.0, 0.0])
    # if delaytype == 'Hyperbolic': theta = (3.5, 3.0)
    # g = los_angeles(theta, delaytype, path=path)[3]

    ## Comparison plot




