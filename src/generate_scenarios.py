#!/usr/bin/env python
import json
import numpy as np
import random

def set_sensors(s, prop, NLP_max, NB_max, NS_max, NL_max):
    s['NLP'] = np.ceil(NLP_max * prop)
    s['NB'] = np.ceil(NB_max * prop)
    s['NS'] = np.ceil(NS_max * prop)
    s['NL'] = np.ceil(NL_max * prop)
    return s

def test_all_sampled(outfile='scenarios_all_sampled.txt'):
    scenarios = []
    scenarios.extend(random.sample(test_all(outfile=None),10))
    scenarios.extend(test_all_links(outfile=None, n=10))
    scenarios.extend(test_least_squares(outfile=None,n=10))

    for s in scenarios:
        check_scenario(s)

    if outfile is not None:
        dump(scenarios,outfile)

    return scenarios

def test_all_links(outfile='scenarios_all_links.txt',n=100):
    """
    Test performance of solvers under the condition where all links are observed

    WARNING: results will be different every time this is run (sampling)
    :return:
    """
    scenarios = []

    solver = 'LS' # override solver

    scenarios_all = test_all(outfile=None)
    samples = random.sample(scenarios_all, n) # sample pool of all scenarios

    for s in samples:
        s['solver'] = solver
        s['all_links'] = True
        scenarios.append(s.copy())

    for s in scenarios:
        check_scenario(s)

    if outfile is not None:
        dump(scenarios,outfile)

    return scenarios

def test_least_squares(outfile='scenarios_least_squares.txt',n=100):
    """
    Test performance of various methods (BB/LBFGS/DORE) of the LS solver in
    controlled manners

    WARNING: results will be different every time this is run (sampling)
    :return:
    """
    scenarios = []

    solver = 'LS' # override solver
    methods = ['BB','LBFGS','DORE']
    inits = [True, False]

    scenarios_all = test_all(outfile=None)
    samples = random.sample(scenarios_all, n) # sample pool of all scenarios

    for s in samples:
        s['solver'] = solver
        for method in methods:
            s['method'] = method
            for init in inits:
                s['init'] = init
                scenarios.append(s.copy())

    for s in scenarios:
        check_scenario(s)

    if outfile is not None:
        dump(scenarios,outfile)

    return scenarios

def test_noise():
    fname = 'scenarios_noise.txt'
    pass

def test_basic(iterations=1,proportions=[1],solvers=['LS'],
               nrow_min=3,nrow_max=4,row_step=1,ncol_min=4,ncol_max=5,col_step=1,
               EQ_NLP_max=122,EQ_NB_max=128,EQ_NS_max=128,EQ_NL_max=128):
    scenarios = []

    for i in xrange(iterations):
        s = {}
        s['trial'] = i
        for solver in solvers:
            s['solver'], s['sparse'] = solver, False
            if solver == 'LS':
                # TODO use the results of test_least_squares here
                s['method'], s['init'] = 'BB', False

            models = ['UE','SO','P']

            for model in models:
                s['model'] = model
                # print json.dumps(s, sort_keys=True, indent=4 * ' ')
                # P matrices, scaling cellpath and linkpath sensing
                if model == 'P':
                    nrows = range(nrow_min,nrow_max,row_step)
                    ncols = range(ncol_min,ncol_max,col_step)
                    for nrow in nrows:
                        # FIXME correct for BI limitations
                        if solver == 'BI' and nrow > 4:
                            continue
                        # FIXME correct for BI limitations
                        for ncol in ncols:
                            if solver == 'BI' and ncol > 4:
                                continue
                            s['nrow'], s['ncol'] = nrow, ncol
                            s['nodroutes'] = 15 # FIXME more configurations?
                            NLP_max = (ncol-1) * (nrow-1) * 2 + ncol + nrow - 2
                            NB_max = ncol * nrow * 8
                            NS_max, NL_max = 0, NLP_max
                            for prop in proportions:
                                s = set_sensors(s, prop, NLP_max, NB_max, NS_max, NL_max)
                                for sparse in [True, False]:
                                    s['sparse'] = sparse
                                    scenarios.append(s.copy())

                # UE/SO matrices, scaling cellpath and linkpath sensing
                else:
                    if solver == 'BI' or i >= 1:
                        continue
                    for prop in proportions:
                        s = set_sensors(s, prop, EQ_NLP_max, EQ_NB_max,
                                        EQ_NS_max, EQ_NL_max)
                        s['sparse'] = True if model == 'SO' else False
                        scenarios.append(s.copy())
    return scenarios

def test_all(CS_only=False,outfile='scenarios_all.txt'):
    """
    Mega test of most things

    Note: we have to test CS separately due to its dependence on MATLAB
    :param CS_only:
    :param outfile:
    :return:
    """
    iterations = 5
    solvers = ['CS'] if CS_only else ['LS','BI']
    proportions = [0.1, 0.18, 0.25, 0.33, 0.5, 0.66, 0.75, 0.9, 0.95, 1]
    EQ_NLP_max, EQ_NB_max, EQ_NS_max, EQ_NL_max = 122, 128, 128, 128
    nrow_min, nrow_max, ncol_min, ncol_max, row_step, col_step = 1,11,2,11,3,2

    scenarios = test_basic(iterations=iterations,proportions=proportions,
                           solvers=solvers,nrow_min=nrow_min,nrow_max=nrow_max,
                           row_step=row_step,ncol_min=ncol_min,
                           ncol_max=ncol_max,col_step=col_step,
                           EQ_NLP_max=EQ_NLP_max,EQ_NB_max=EQ_NB_max,
                           EQ_NS_max=EQ_NB_max,EQ_NL_max=EQ_NL_max)

    for s in scenarios:
        check_scenario(s)

    if outfile is not None:
        dump(scenarios,outfile)

    return scenarios

def dump(scenarios, filename):
    with open(filename,'w') as out:
        for s in scenarios:
            out.write('%s\n' % json.dumps(s))

def check_scenario(s):
    check_keys(s,['solver','model','sparse','NLP','NB','NS','NL'])
    if s['model'] == 'P':
        check_keys(s,['nrow','ncol','nodroutes'])
    if s['solver'] == 'LS':
        check_keys(s,['method','init'])

def check_keys(d,keys):
    for key in keys:
        if key not in d:
            return "Missing %s in %s" % (key, d)

def test_small(outfile='scenarios_small.txt'):
    """
    For testing that hadoop setup is working
    :return:
    """
    scenarios = []
    iterations = 2
    solvers = ['LS']
    proportions = [0.5, 1]
    nrow_min, nrow_max, ncol_min, ncol_max, row_step, col_step = 3,4,4,5,1,1
    EQ_NLP_max, EQ_NB_max, EQ_NS_max, EQ_NL_max = 122, 128, 128, 128

    scenarios = test_basic(iterations=iterations,proportions=proportions,
                           solvers=solvers,nrow_min=nrow_min,nrow_max=nrow_max,
                           row_step=row_step,ncol_min=ncol_min,
                           ncol_max=ncol_max,col_step=col_step,
                           EQ_NLP_max=EQ_NLP_max,EQ_NB_max=EQ_NB_max,
                           EQ_NS_max=EQ_NB_max,EQ_NL_max=EQ_NL_max)

    for s in scenarios:
        check_scenario(s)

    if outfile is not None:
        dump(scenarios,outfile)

    return scenarios

if __name__ == "__main__":
    test_small()
    test_least_squares()
    test_all()
    test_all_links()
    test_all_sampled()
