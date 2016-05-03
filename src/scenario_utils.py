from __future__ import division
import ipdb
import argparse
import logging
import time
from random import randint
import cPickle as pickle
from cPickle import BadPickleGet
import json

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

from sklearn.isotonic import IsotonicRegression

import config as c
# from python.isotonic_regression.simplex_projection import simplex_projection
# from projection import pysimplex_projection
from BSLS.python.bsls_utils import  x2z, particular_x0
from BSLS.python.isotonic_regression.block_isotonic_regression import block_isotonic_regression
from BSLS.python.gradient_descent import GradientDescent


##############################################################################
# Parameter handling
##############################################################################
def args_from_TN(TN, args=None):
    # TrafficNetwork args
    if args is None:
        args = {}

    if TN.__class__.__name__ == 'EquilibriumNetwork':
        args['model'] = 'SO' if TN.SO is True else 'UE'
    elif TN.__class__.__name__ == 'GridNetwork':
        args['model'] = 'P'
        args['nrow'] = TN.m
        args['ncol'] = TN.n
        args['sparse'] = True if TN.concentration is None else False
        args['nodroutes'] = TN.r
    return args

def args_from_SC(SC, args=None):
    # Sensor configuration args
    if args is None:
        args = {}
    args['NS'] = int(SC.num_cellpath_NS)
    # FIXME untested and not integrated well, but also unused for the time being
    args['NL'] = min(int(SC.num_cellpath_NL), SC.num_link)
    args['NB'] = int(SC.num_cellpath_NB)
    args['NLP'] = len(SC.linkpath_sensors) if SC.linkpath_sensors is \
                                                    not None else 0
    return args

def args_from_solver(solver, args=None):
    # Sensor configuration args
    if args is None:
        args = {}

    if solver.__class__.__name__ == 'SolverLS':
        args['solver'] = 'LS'
        args['method'] = solver.method
        args['init'] = solver.init
        args['noise'] = solver.noise
        args['eq'] = solver.eq
    elif solver.__class__.__name__ == 'SolverLSQR':
        args['solver'] = 'LSQR'
        args['damp'] = solver.damp
        args['eq'] = solver.eq
        args['noise'] = 0.0
        args['init'] = False
    elif solver.__class__.__name__ == 'SolverBI':
        args['solver'] = 'BI'
        args['sparse_BI'] = solver.sparse
        args['noise'] = 0.0
        args['eq'] = 'CP'
        args['init'] = False
    elif solver.__class__.__name__ == 'SolverCS':
        args['solver'] = 'CS'
        args['noise'] = solver.noise
        args['eq'] = solver.eq
        args['init'] = solver.init
        args['method'] = solver.method
        args['iters'] = solver.iters
    args['all_links'] = solver.full
    args['use_L'] = solver.L
    args['use_OD'] = solver.OD
    args['use_CP'] = solver.CP
    args['use_LP'] = solver.LP
    return args

def new_s(s=None):
    base_s = vars(new_namespace())
    if s is None:
        return base_s
    for (k,v) in base_s.iteritems():
        if k not in s:
            s[k] = v
    return s

def new_namespace():
    ns = argparse.Namespace()
    ns.solver = 'LS'
    ns.model = 'P'
    ns.sparse = False
    ns.noise = 0.0
    ns.all_links = True
    ns.use_L = True
    ns.use_OD = True
    ns.use_CP = True
    ns.use_LP = True
    ns.NLP = 0
    ns.NB = 0
    ns.NS = 0
    ns.NL = 0
    ns.eq = 'CP'
    ns.init = False

    ns.output = False
    ns.iters = 0
    ns.nodroutes = 15
    ns.damp = 0.0
    ns.alpha = 1.0
    ns.nrow = 0
    ns.ncol = 0
    ns.method = ''
    return ns

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

    # BI solver only
    parser.add_argument('--alpha',dest='alpha',type=float,default=1,
                        help='BI only: spread')

    # LSQR solver only
    parser.add_argument('--damp',dest='damp',type=float,default=0.0,
                        help='LSQR only: damping factor')

    # CS solver only
    parser.add_argument('--iters',dest='iters',type=int,default=6000,
                        help='CS only: # iterations for sampling')

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

    if args.solver == 'CS':
        args.iters = int(params['iters'])

    return args


##############################################################################
# Misc utilities
##############################################################################
def load(fname=None):
    try:
        with open(fname) as f:
            return pickle.load(f)
    except (IOError, ValueError, BadPickleGet):
        print 'Error loading %s' % fname
        return None

def save(x, fname=None, prefix=None):
    if fname is None and prefix is not None:
        t = int(time.time())
        r = randint(1e5,999999)
        fname = "%s_%s_%s.pkl" % (prefix, t, r)
    if fname is not None:
        with open(fname, 'w') as f:
            pickle.dump(x, f)

class NumpyAwareJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray) and obj.ndim == 1:
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


##############################################################################
# Solver helpers
##############################################################################
def CS_solve(A,b,x0,N,block_sizes,mask):
    z0 = x2z(x0,block_sizes)
    target = A.dot(x0)-b
    xp = particular_x0(block_sizes)

    AT = A.T.tocsr()
    NT = N.T.tocsr()
    mind = mask.nonzero()[0]

    assert np.all((xp + N.dot(x2z(mask,block_sizes)))[mind])==True, \
        'Faulty conversion to and from x'

    eps = 0
    gamma = 1
    def f(z):
        sparse_term = np.nan_to_num(1/((xp+N.dot(z))[mind]))
        sparse_term = np.minimum(sparse_term, 1e5)
        sparse_term = np.maximum(sparse_term, -1e5)
        return np.sum(sparse_term) + gamma*0.5*la.norm(A.dot(N.dot(z))+target)**2

    def nabla_f(z):
        sparse_term = np.zeros(mask.size)
        sparse_term[mind] = -np.nan_to_num(1/(xp + N.dot(z))[mind]**2)
        if np.any(np.abs(sparse_term) > 1e300):
            sparse_term = np.minimum(sparse_term, 1e5)
            sparse_term = np.maximum(sparse_term, -1e5)
            logging.error("Sparse term near infinity")
            # ipdb.set_trace()
        else:
            print 'good'
        return NT.dot(sparse_term + gamma*AT.dot(A.dot(N.dot(z)) + target))

    ir = IsotonicRegression(y_min=0, y_max=1)
    cum_blocks = np.concatenate(([0], np.cumsum(block_sizes-1)))
    blocks_start = cum_blocks
    blocks_end = cum_blocks[1:]

    def proj(x):
        return block_isotonic_regression(x, ir, block_sizes, blocks_start,
                                         blocks_end)
        # value = simplex_projection(block_sizes - 1,x)
        # value = pysimplex_projection(block_sizes - 1,x)
        # return projected_value

    gd = GradientDescent(z0=z0, f=f, nabla_f=nabla_f, proj=proj, method='BB',
                         A=A, N=N, target=target)
    iters, times, states = gd.run()
    x = xp + N.dot(states[-1])
    assert np.all(x >= 0), 'x shouldn\'t have negative entries after projection'
    print f(z0), f(states[-1])
    ipdb.set_trace()
    return x, f(z0), f(states[-1])

def solve_in_z(A,b,x0,N,block_sizes,method):
    if block_sizes is not None and len(block_sizes) == A.shape[1]:
        logging.error('Trivial example: nblocks == nroutes, exiting solver')
        import sys
        sys.exit()

    z0 = x2z(x0,block_sizes)
    target = A.dot(x0)-b

    AT = A.T.tocsr()
    NT = N.T.tocsr()

    f = lambda z: 0.5 * la.norm(A.dot(N.dot(z)) + target)**2
    nabla_f = lambda z: NT.dot(AT.dot(A.dot(N.dot(z)) + target))

    ir = IsotonicRegression(y_min=0, y_max=1)
    cum_blocks = np.concatenate(([0], np.cumsum(block_sizes-1)))
    blocks_start = cum_blocks
    blocks_end = cum_blocks[1:]

    def proj(x):
        return block_isotonic_regression(x, ir, block_sizes, blocks_start,
                                         blocks_end)
        # value = simplex_projection(block_sizes - 1,x)
        # value = pysimplex_projection(block_sizes - 1,x)
        # return projected_value

    if method == 'DORE':
        gd = GradientDescent(z0=z0, f=f, nabla_f=nabla_f, proj=proj,
                             method=method, A=A, N=N, target=target)
    else:
        gd = GradientDescent(z0=z0, f=f, nabla_f=nabla_f, proj=proj,
                             method=method)
    iters, times, states = gd.run()
    x = particular_x0(block_sizes) + N.dot(states[-1])
    assert np.all(x >= 0), 'x shouldn\'t have negative entries after projection'
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
        xp = particular_x0(block_sizes)
        x_hat = N.dot(np.array(states).T) + np.tile(xp,(d,1)).T
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
    if type(error) == np.float:
        error = np.array([error])
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

