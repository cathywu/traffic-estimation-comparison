import ipdb
import time

from pprint import pprint
import logging

import config as c

# Dependencies for data generation
from networks.EquilibriumNetwork import EquilibriumNetwork
from networks.GridNetwork import GridNetwork
from sensors.SensorConfiguration import SensorConfiguration

# Import solvers
from SolverLS import SolverLS
from SolverBI import SolverBI
from SolverCS import SolverCS
from SolverLSQR import SolverLSQR

# Dependencies for Bayesian inference
from grid_model import load_model, create_model
from grid_simulation import MCMC

# Dependencies for least squares
from python.util import load_data, solver_input

# Dependencies for compressed sensing

# Dependencies for traffic assignment
import scipy.io
import numpy as np

from isttt2014_experiments import synthetic_data
from linkpath import LinkPath
import path_solver
import Waypoints as WP

from synth_utils import to_sp, array
from scenario_utils import parser, update_args
from synth_utils import deprecated

class Scenario:
    def __init__(self, TN, S, solver):
        self.TN = TN
        self.S = S
        self.solver = solver
        self.output = None

    def run(self):
        self.output = self.solver.solve()

# Data generation
# -------------------------------------
def generate_data_P(nrow=5, ncol=6, nodroutes=4, nnz_oroutes=10,
                    NB=60, NS=20, NL=15, NLP=98, type='small_graph_OD.mat'):
    """
    Generate and export probabilistic matrices
    """
    prefix = '%s/' % c.DATA_DIR
    data = static_matrix.export_matrices(prefix, nrow, ncol, nodroutes=nodroutes,
                                  nnz_oroutes=nnz_oroutes, NB=NB, NS=NS, NL=NL,
                                  NLP=NLP, export=False, type=type)
    return data
    # FIXME return data

def generate_data_UE(data=None, export=False, SO=False, demand=3, N=30,
                     withODs=False, NLP=122):
    """
    Generate and export UE matrices
    """
    # FIXME copy mat file to local directory?
    path='hadoop/los_angeles_data_2.mat'
    g, x_true, l, path_wps, wp_trajs, obs, wp = synthetic_data(data, SO, demand, N,
                                                           path=path,fast=False)
    x_true = array(x_true)
    obs=obs[0]
    A_full = path_solver.linkpath_incidence(g)
    A = to_sp(A_full[obs,:])
    A_full = to_sp(A_full)
    U,f = WP.simplex(g, wp_trajs, withODs)
    T,d = path_solver.simplex(g)

    data = {'A_full': A_full, 'b_full': A_full.dot(x_true),
            'A': A, 'b': A.dot(x_true), 'x_true': x_true,
            'T': to_sp(T), 'd': array(d),
            'U': to_sp(U), 'f': array(f) }

    if NLP is not None:
        lp = LinkPath(g,x_true,N=NLP)
        lp.update_lp_flows()
        V,g = lp.simplex_lp()
        data['V'], data['g'] = V, g

    # Export
    if export:
        if not SO:
            fname = '%s/UE_graph.mat' % c.DATA_DIR
        else:
            fname = '%s/SO_graph.mat' % c.DATA_DIR
        scipy.io.savemat(fname, data, oned_as='column')

    return data

# Experimentation helper functions
# -------------------------------------
@deprecated
def experiment_BI(sparse, full=False, L=True, OD=True, CP=True, LP=True, data=None):
    """
    Bayesian inference experiment
    """
    AA, bb_obs, EQ, x_true, scaling, out = solver_input(data, full=full, L=L,
                                OD=OD, CP=CP, LP=LP, eq='CP', EQ_elim=False)
    output = out
    if EQ is None:
        output['error'] = "EQ constraint matrix is empty"
    model,alpha,x_pri = create_model(AA, bb_obs, EQ, x_true, sparse=sparse)
    output['alpha'] = alpha

    # model = create_model('%s/%s' % (c.DATA_DIR,test),sparse)
    if np.all(x_pri==1):
        x_last, error, output = LS_postprocess([x_pri], x_pri, AA.todense(), bb_obs,
                                           x_true, output=output, is_x=True)
    else:
        model, trace, init_time, duration = MCMC(model)
        output['init_time'], output['duration'] = init_time, duration

        x_blocks = None
        for varname in sorted(trace.varnames):
            # flatten the trace and normalize
            if trace.get_values(varname).shape[1] == 0:
                continue
            x_block = np.array([x/sum(x) for x in trace.get_values(varname)])
            if x_blocks is not None:
                x_blocks = np.hstack((x_blocks, x_block))
            else:
                x_blocks = x_block

        x_last, error, output = LS_postprocess(x_blocks, x_blocks[0,:], AA.todense(),
                                    bb_obs, x_true, output=output, is_x=True)
    output['blocks'] = EQ.shape[0] if EQ is not None else None
    return output

def experiment_TA():
    pass

@deprecated
def experiment_CS(args, test='temp', full=False, L=True, OD=True, CP=True, LP=True,
                  eq='CP', data=None, init=False):
    # CS test config
    CS_PATH = '/Users/cathywu/Dropbox/Fa13/EE227BT/traffic-project'
    OUT_PATH = '%s/data/output-cathywu/' % CS_PATH

    # Test parameters
    alg = 'cvx_random_sampling_L1_30_replace'
    # alg = 'cvx_oracle'
    # alg = 'cvx_unconstrained_L1'
    # alg = 'cvx_L2'
    # alg = 'cvx_raw'
    # alg = 'cvx_weighted_L1'
    # alg = 'cvx_hot_start_lp'
    # alg = 'cvx_single_block_L_infty'
    # alg = 'cvx_random_sample_L_infty'
    # alg = 'cvx_mult_blocks_L_infty'
    # alg = 'cvx_block_descent_L_infty'
    # alg = 'cvx_entropy'

    # Load test and export to .mat
    init_time = time.time()
    if data is None and test is not None:
        fname = '%s/%s' % (c.DATA_DIR,test)
        A, b, N, block_sizes, x_true, nz, flow, rsort_index, x0, out = \
            load_data(fname, full=full, L=L, OD=OD, CP=CP, LP=LP, eq=eq,
                      init=init)
    else:
        A, b, N, block_sizes, x_true, nz, flow, rsort_index, x0, out = \
            solver_input(data, full=full, L=L, OD=OD, CP=CP, LP=LP,
                         eq=eq, init=init)
    output = out

    if block_sizes is None or N is None:
        output['error'] = "No EQ constraint"
        return output

    # Perturb
    if args.noise:
        b_true = b
        delta = np.random.normal(scale=b*args.noise)
        b = b + delta

    fname = '%s/CS_%s' % (c.DATA_DIR,test)
    try:
        scipy.io.savemat(fname, { 'A': A, 'b': b, 'x_true': x_true, 'flow' : flow,
                              'x0': x0, 'block_sizes': block_sizes},
                     oned_as='column')
    except TypeError:
        pprint({ 'A': A, 'b': b, 'x_true': x_true, 'flow' : flow,
                              'x0': x0, 'block_sizes': block_sizes })
        ipdb.set_trace()

    # Perform test via MATLAB
    from pymatbridge import Matlab
    mlab = Matlab()
    mlab.start()
    duration_time = time.time()
    mlab.run_code('cvx_solver mosek;')
    mlab.run_code("addpath '~/mosek/7/toolbox/r2012a';")

    init_time = time.time() - init_time
    output['init_time'] = init_time

    p = mlab.run_func('%s/scenario_to_output.m' % CS_PATH,
                      { 'filename' : fname, 'type' : test, 'algorithm' : alg,
                        'outpath' : OUT_PATH })
    duration_time = time.time() - duration_time
    mlab.stop()
    x = array(p['result'])

    _, _, output = LS_postprocess([x],x,A,b,x_true,block_sizes=block_sizes,
                                  output=output,is_x=True)
    output['duration'], output['iters'], output['times'] = duration_time, [0], [0]

    return output

@deprecated
def experiment_LSQR(args, test=None, data=None, full=False, L=True, OD=True,
                    CP=True, LP=True, damp=0.0):
    init_time = time.time()
    A, b, x0, x_true, out = solver_input(data, full=full, L=L, OD=OD, CP=CP,
                                         LP=LP, solve=True, damp=damp)
    init_time = time.time() - init_time
    output = out
    output['init_time'] = init_time

    if A is None:
        output['error'] = "Empty objective"
        return output

    _, _, output = LS_postprocess([x0],x0,A,b,x_true,output=output,is_x=True)

    output['duration'], output['iters'], output['times'] = 0, [0], [0]
    return output

@deprecated
def experiment_LS(args, test=None, data=None, full=True, L=True, OD=True,
                  CP=True, LP=True, eq='CP', init=True):
    """
    Least squares experiment
    :param test:
    :return:
    """
    ## LS experiment
    ## TODO: invoke solver
    init_time = time.time()
    if data is None and test is not None:
        fname = '%s/%s' % (c.DATA_DIR,test)
        A, b, N, block_sizes, x_true, nz, flow, rsort_index, x0, out = \
            load_data(fname, full=full, L=L, OD=OD, CP=CP, LP=LP, eq=eq,
                      init=init)
    else:
        A, b, N, block_sizes, x_true, nz, flow, rsort_index, x0, out = \
            solver_input(data, full=full, L=L, OD=OD, CP=CP, LP=LP,
                         eq=eq, init=init)
    init_time = time.time() - init_time
    output = out
    output['init_time'] = init_time

    # x0 = np.array(util.block_e(block_sizes - 1, block_sizes))

    if args.noise:
        b_true = b
        delta = np.random.normal(scale=b*args.noise)
        b = b + delta

    if block_sizes is not None:
        logging.debug("Blocks: %s" % block_sizes.shape)
    # z0 = np.zeros(N.shape[1])
    if N is None or (block_sizes-1).any() == False:
        iters, times, states = [0],[0],[x0]
        x_last, error, output = LS_postprocess(states,x0,A,b,x_true,scaling=flow,
                                               block_sizes=block_sizes,N=N,
                                               output=output,is_x=True)
    else:
        iters, times, states = LS_solve(A,b,x0,N,block_sizes,args)
        x_last, error, output = LS_postprocess(states,x0,A,b,x_true,scaling=flow,
                                               block_sizes=block_sizes,N=N,
                                               output=output)

    # LS_plot(x_last, times, error)
    output['duration'] = np.sum(times)

    output['iters'], output['times'] = list(iters), list(times)
    return output

def scenario(params=None, log='INFO'):
    # use argparse object as default template
    p = parser()
    args = p.parse_args()
    if args.log in c.ACCEPTED_LOG_LEVELS:
        logging.basicConfig(level=eval('logging.'+args.log))
    if params is not None:
        args = update_args(args, params)

    print args

    if args.model == 'P':
        type = 'small_graph_OD.mat' if args.sparse else 'small_graph_OD_dense.mat'

        data = generate_data_P(nrow=args.nrow, ncol=args.ncol,
                               nodroutes=args.nodroutes,
                               NB=args.NB, NL=args.NL, NLP=args.NLP,
                               type=type)
        if 'error' in data:
            return {'error' : data['error']}
    else:
        SO = True if args.model == 'SO' else False
        # N0, N1, scale, regions, res, margin
        config = (args.NB, args.NL, 0.2, [((3.5, 0.5, 6.5, 3.0), args.NS)], (6,3), 2.0)
        # data[0] = (20, 40, 0.2, [((3.5, 0.5, 6.5, 3.0), 20)], (12,6), 2.0)
        # data[1] = (10, 20, 0.2, [((3.5, 0.5, 6.5, 3.0), 10)], (10,5), 2.0)
        # data[2] = (5, 10, 0.2, [((3.5, 0.5, 6.5, 3.0), 5)], (6,3), 2.0)
        # data[3] = (3, 5, 0.2, [((3.5, 0.5, 6.5, 3.0), 2)], (4,2), 2.0)
        # data[4] = (1, 3, 0.2, [((3.5, 0.5, 6.5, 3.0), 1)], (2,2), 2.0)
        # TODO trials?
        data = generate_data_UE(data=config, SO=SO, NLP=args.NLP)
        if 'error' in data:
            return {'error' : data['error']}

    eq = 'CP' if args.use_CP else 'OD'
    if args.solver == 'CS':
        sol = SolverCS(args, full=args.all_links, L=args.use_L,
                    OD=args.use_OD, CP=args.use_CP, LP=args.use_LP, eq=eq, data=data)
    elif args.solver == 'BI':
        sol = SolverBI(args.sparse, full=args.all_links, L=args.use_L,
                    OD=args.use_OD, CP=args.use_CP, LP=args.use_LP, data=data)
    elif args.solver == 'LS':
        sol = SolverLS(args, full=args.all_links, init=args.init,
                    L=args.use_L, OD=args.use_OD, CP=args.use_CP,
                    LP=args.use_LP, eq=eq, data=data)
    elif args.solver == 'LSQR':
        sol = SolverLSQR(args, full=args.all_links, L=args.use_L,
                    OD=args.use_OD, CP=args.use_CP, LP=args.use_LP, data=data)
    sol.setup()
    sol.solve()
    sol.analyze()

    if args.output == True:
        pprint(sol.output)

    return sol.output

if __name__ == "__main__":
    output = scenario()

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




