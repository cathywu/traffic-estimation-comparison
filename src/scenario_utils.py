import ipdb
import argparse
import logging
import time
from random import randint

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

import config as c
from BSC_NNLS.python import util
from BSC_NNLS.python.c_extensions.simplex_projection import simplex_projection
from BSC_NNLS.python import BB, LBFGS, DORE, solvers

def load(fname=None):
    import pickle
    try:
        with open(fname, 'w') as f:
            return pickle.load(f)
    except IOError:
        return None

def save(x, fname=None, prefix=None):
    import pickle
    if fname is None and prefix is not None:
        t = int(time.time())
        r = randint(1e5,999999)
        fname = "%s_%s_%s.pkl" % (prefix, t, r)
    if fname is not None:
        with open(fname, 'w') as f:
            pickle.dump(x, f)

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', dest='log', nargs='?', const='INFO',
                        default='WARN', help='Set log level (default: WARN)')
    parser.add_argument('--output',dest='output',action='store_true',
                        default=False,help='Print output toggle (false)')

    parser.add_argument('--solver',dest='solver',type=str,default='LS',
                        help='Solver name') # CS/BI/LS
    parser.add_argument('--model',dest='model',type=str,default='P',
                        help='Macro traffic dynamics model') # P/UE/SO
    parser.add_argument('--noise',dest='noise',type=float,default=None,
                        help='Noise level')
    parser.add_argument('--all_links',dest='all_links',action='store_true',
                        default=False,help='All links observed (false)')

    # Sensor toggles
    parser.add_argument('--use_L',dest='use_L',action='store_false',
                        default=True,help='Use L sensors (true)')
    parser.add_argument('--use_OD',dest='use_OD',action='store_false',
                        default=True,help='Use OD sensors (true)')
    parser.add_argument('--use_CP',dest='use_CP',action='store_false',
                        default=True,help='Use CP sensors (true)')
    parser.add_argument('--use_LP',dest='use_LP',action='store_false',
                        default=True,help='Use LP sensors (true)')

    # LS solver only
    parser.add_argument('--method',dest='method',type=str,default='BB',
                        help='LS only: Least squares method')
    parser.add_argument('--init',dest='init',action='store_true',
                        default=False,help='LS only: initial solution from data')

    # LSQR solver only
    parser.add_argument('--damp',dest='damp',type=float,default=0.0,
                        help='LSQR only: damping factor')

    # Sparsity
    parser.add_argument('--sparse',dest='sparse',action='store_true',
                        default=False,help='BI/P only: Sparse toggle for route flow sampling')

    # Linkpath and cellpath sensing
    parser.add_argument('--NLP',dest='NLP',type=int,default=9,
                        help='Number of linkpath sensors (sampled uniformly)')
    parser.add_argument('--NB',dest='NB',type=int,default=48,
                        help='Number of cells sampled uniformly in bbox')
    parser.add_argument('--NS',dest='NS',type=int,default=0,
                        help='Number of cells sampled uniformly in region')
    parser.add_argument('--NL',dest='NL',type=int,default=9,
                        help='Number of cells sampled uniformly in links')

    # P model only
    parser.add_argument('--nrow',dest='nrow',type=int,default=3,
                        help='P only: Number of rows in grid network')
    parser.add_argument('--ncol',dest='ncol',type=int,default=4,
                        help='P only: Number of rows in grid network')
    parser.add_argument('--nodroutes',dest='nodroutes',type=int,default=15,
                        help='P only: Number of routes per OD pair')

    return parser

def update_args(args, params):
    """
    Update argparse object with attributes from params dictionary
    :param args:
    :param params:
    :return:
    """
    args.solver = params['solver'] # LS/BI/CS
    args.model = params['model'] # P/UE/SO
    args.sparse = bool(params['sparse']) # sparse toggle for route flow sampling
    if 'all_links' in params:
        args.all_links = bool(params['all_links'])

    # Sensor toggles
    args.use_L = bool(params['use_L']) if 'use_L' in params else True
    args.use_OD = bool(params['use_OD']) if 'use_OD' in params else True
    args.use_CP = bool(params['use_CP']) if 'use_CP' in params else True
    args.use_LP = bool(params['use_LP']) if 'use_LP' in params else True

    # Sensor configurations
    args.NLP = int(params['NLP']) # number of linkpath sensors (sampled randomly)
    args.NB = int(params['NB'])   # number of cells sampled uniformly in bbox
    args.NS = int(params['NS'])   # number of cells sampled uniformly in region
    args.NL = int(params['NL'])   # number of cells sampled uniformly in links

    if args.model == 'P':
        # For --model P only:
        args.nrow = int(params['nrow']) # number of rows in grid network
        args.ncol = int(params['ncol']) # number of cols in grid network
        args.nodroutes = int(params['nodroutes']) # number of routes per OD pair

    if args.solver == 'LS':
        # For --solver LS only:
        args.method = params['method'] # BB/LBFGS/DORE
        args.init = bool(params['init']) # True/False

    if args.solver == 'LSQR':
        args.damp = float(params['damp'])

    return args

def LS_solve(A,b,x0,N,block_sizes,method):
    z0 = util.x2z(x0,block_sizes)
    target = A.dot(x0)-b

    options = { 'max_iter': 300000,
                'verbose': 1,
                'opt_tol' : 1e-30,
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

    import time
    iters, times, states = [], [], []
    def log(iter_,state,duration):
        iters.append(iter_)
        times.append(duration)
        states.append(state)
        start = time.time()
        return start

    logging.debug('Starting %s solver...' % method)
    if method == 'LBFGS':
        LBFGS.solve(z0+1, f, nabla_f, solvers.stopping, log=log,proj=proj,
                    options=options)
        logging.debug("Took %s time" % str(np.sum(times)))
    elif method == 'BB':
        BB.solve(z0,f,nabla_f,solvers.stopping,log=log,proj=proj,
                 options=options)
    elif method == 'DORE':
        # setup for DORE
        alpha = 0.99
        lsv = util.lsv_operator(A, N)
        logging.info("Largest singular value: %s" % lsv)
        A_dore = A*alpha/lsv
        target_dore = target*alpha/lsv

        DORE.solve(z0, lambda z: A_dore.dot(N.dot(z)),
                   lambda b: N.T.dot(A_dore.T.dot(b)), target_dore, proj=proj,
                   log=log,options=options,record_every=100)
        A_dore = None
    logging.debug('Stopping %s solver...' % method)
    return iters, times, states

def LS_postprocess(states, x0, A, b, x_true, scaling=None, block_sizes=None,
                   output=None, N=None, is_x=False):
    if x_true is None:
        return [], [], output
    if scaling is None:
        scaling = np.ones(x_true.shape)
    if output is None:
        output = {}
    d = len(states)

    # Convert back to x (from z) if necessary
    if not is_x and N.size > 0:
        x_hat = N.dot(np.array(states).T) + np.tile(x0,(d,1)).T
    else:
        x_hat = np.array(states).T
    x_last = x_hat[:,-1]
    n = x_hat.shape[1]

    # Record sizes
    output['AA'] = A.shape
    output['x_hat'] = x_hat.shape
    output['blocks'] = block_sizes.shape if block_sizes is not None else None

    # Objective error, i.e. 0.5||Ax-b||_2^2
    starting_error = 0.5 * la.norm(A.dot(x0)-b)**2
    opt_error = 0.5 * la.norm(A.dot(x_true)-b)**2
    diff = A.dot(x_hat) - np.tile(b,(d,1)).T
    error = 0.5 * np.diag(diff.T.dot(diff))
    output['0.5norm(Ax-b)^2'], output['0.5norm(Ax_init-b)^2'] = error, starting_error
    output['0.5norm(Ax*-b)^2'] = opt_error

    # Route flow error, i.e ||x-x*||_1
    x_true_block = np.tile(x_true,(n,1))
    x_diff = x_true_block-x_hat.T

    scaling_block = np.tile(scaling,(n,1))
    x_diff_scaled = scaling_block * x_diff
    x_true_scaled = scaling_block * x_true_block

    # most incorrect entry (route flow)
    dist_from_true = np.max(x_diff_scaled,axis=1)
    output['max|f * (x-x_true)|'] = dist_from_true

    # num incorrect entries
    wrong = np.bincount(np.where(x_diff > 1e-3)[0])
    output['incorrect x entries'] = wrong

    # % route flow error
    per_flow = np.sum(np.abs(x_diff_scaled), axis=1) / np.sum(x_true_scaled, axis=1)
    output['percent flow allocated incorrectly'] = per_flow

    # initial route flow error
    start_dist_from_true = np.max(scaling * np.abs(x_true-x0))
    output['max|f * (x_init-x_true)|'] = start_dist_from_true

    return x_last, error, output

def LS_plot(x_last, times, error):
    plt.figure()
    plt.hist(x_last)

    plt.figure()
    plt.loglog(np.cumsum(times),error)
    plt.show()

