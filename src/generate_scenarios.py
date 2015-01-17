#!/usr/bin/env python
import json
import numpy as np
import random

def new_s():
    return {'solver': 'LS', 'model': 'P', 'sparse': False, 'noise': 0.0,
            'all_links': False, 'use_L': True, 'use_OD': True, 'use_CP': True,
            'use_LP': True, 'NLP': 0, 'NB': 0, 'NS': 0, 'NL': 0}

def chunk(l, n):
    """ Yield successive n-sized chunks from l.
    """
    for i in xrange(0, len(l), n):
        yield l[i:i+n]

def set_sensors(s, prop, NLP_max=None, NB_max=None, NS_max=None, NL_max=None):
    s['NLP'] = np.ceil(NLP_max * prop) if NLP_max is not None else 0
    s['NB'] = np.ceil(NB_max * prop) if NB_max is not None else 0
    s['NS'] = np.ceil(NS_max * prop) if NS_max is not None else 0
    s['NL'] = np.ceil(NL_max * prop) if NL_max is not None else 0
    return s

def dump(scenarios, filename):
    dir = 'hadoop/input'
    with open('%s/%s' % (dir,filename),'w') as out:
        for s in scenarios:
            out.write('%s\n' % json.dumps(s))

def check_scenario(s):
    check_keys(s,['solver','model','sparse','NLP','NB','NS','NL','use_L','use_OD',
                  'use_CP','use_LP'])
    if s['model'] == 'P':
        check_keys(s,['nrow','ncol','nodroutes'])
    if s['solver'] == 'LS':
        check_keys(s,['method','init'])
    if s['solver'] == 'LSQR':
        check_keys(s,['damp'])

def check_keys(d,keys):
    for key in keys:
        if key not in d:
            return "Missing %s in %s" % (key, d)

def test_all_reduced_reduced_sensors(outfile='scenarios_all_reduced_reduced_sensors.txt'):
    scenarios_temp = []
    scenarios_temp.extend(random.sample(test_all(outfile=None),500))
    scenarios_temp.extend(test_least_squares(outfile=None,n=500))

    scenarios = []
    for scenario in scenarios_temp:
        s = scenario.copy()
        s['NLP'] /= 4
        s['NB'] /= 10
        s['NL'] /= 10
        scenarios.append(s.copy())
        s['NB'] /= 12
        s['NL'] /= 12
        scenarios.append(s.copy())
        s['NB'] /= 14
        s['NL'] /= 14
        scenarios.append(s.copy())
        s['NB'] /= 16
        s['NL'] /= 16
        scenarios.append(s.copy())
        s['NLP'] /= 2
        s['NB'] /= 10
        s['NL'] /= 10
        scenarios.append(s.copy())
        s['NB'] /= 12
        s['NL'] /= 12
        scenarios.append(s.copy())
        s['NB'] /= 14
        s['NL'] /= 14
        scenarios.append(s.copy())
        s['NB'] /= 16
        s['NL'] /= 16
        scenarios.append(s.copy())

    for s in scenarios:
        check_scenario(s)

    if outfile is not None:
        dump(scenarios,outfile)

    return scenarios

def test_all_reduced_sensors(outfile='scenarios_all_reduced_sensors.txt'):
    scenarios_temp = []
    scenarios_temp.extend(random.sample(test_all(outfile=None),500))
    scenarios_temp.extend(test_all_links(outfile=None, n=500))
    scenarios_temp.extend(test_least_squares(outfile=None,n=500))

    scenarios = []
    for scenario in scenarios_temp:
        s = scenario.copy()
        s['NLP'] /= 2
        scenarios.append(s.copy())
        s['NLP'] /= 4
        scenarios.append(s.copy())
        s['NB'] /= 4
        s['NL'] /= 4
        scenarios.append(s.copy())
        s['NB'] /= 8
        s['NL'] /= 8
        scenarios.append(s.copy())

    for s in scenarios:
        check_scenario(s)

    if outfile is not None:
        dump(scenarios,outfile)

    return scenarios

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

def test_bayesian_inference(outfile='scenarios_bayesian_inference.1.txt'):
    iterations = 1
    solvers = ['BI']
    proportions = [0.25, 0.5, 1]
    EQ_NLP_max, EQ_NB_max, EQ_NS_max, EQ_NL_max = 20, 20, 0, 0
    nrow_min, nrow_max, ncol_min, ncol_max, row_step, col_step = 1,5,2,5,1,1

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

def test_all_links_UE(outfile='scenarios_all_links_UE.txt',n=220,method='BB',v=None):
    """
    Test performance of solvers under the condition where all links are observed
    and no other sensors (no linkpath or cellpath sensors). This tests the
    limitations of link sensors.

    WARNING: results will be different every time this is run (sampling)
    :return:
    """
    if v is not None:
        outfile = "%s.%s" % (outfile,v)

    scenarios = []

    solver = 'LS' # override solver

    scenarios_all = test_UE(outfile=None)
    samples = random.sample(scenarios_all, n) # sample pool of all scenarios

    for s in samples:
        s['solver'] = solver
        s['all_links'] = True
        s['NLP'], s['NL'], s['NB'], s['NS'] = 0,0,0,0
        s['method'] = method
        scenarios.append(s.copy())

    for s in scenarios:
        check_scenario(s)

    if outfile is not None:
        dump(scenarios,outfile)

    return scenarios

def test_all_links(outfile='scenarios_all_links.txt',n=500,method='BB',v=None):
    """
    Test performance of solvers under the condition where all links are observed
    and no other sensors (no linkpath or cellpath sensors). This tests the
    limitations of link sensors.

    WARNING: results will be different every time this is run (sampling)
    :return:
    """
    if v is not None:
        outfile = "%s.%s" % (outfile,v)

    scenarios = []

    solver = 'LS' # override solver

    scenarios_all = test_all(outfile=None)
    samples = random.sample(scenarios_all, n) # sample pool of all scenarios

    for s in samples:
        s['solver'] = solver
        s['all_links'] = True
        s['NLP'], s['NL'], s['NB'], s['NS'] = 0,0,0,0
        s['method'] = method
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

def test_LSQR(outfile='scenarios_LSQR_d%0.2f.%d.txt',N=10,damp=0.0):
    import random
    scenarios = []

    solver = 'LSQR'

    sensor_configs = [(True,True,True,True), (True,True,False,False),
        (False,False,True,True),(False,False,True,False),
        (False,False,False,True),(True,False,False,False),
        (False,True,False,False)]
    scenarios_all = test_all(outfile=None)

    for s in scenarios_all:
        s['solver'] = solver
        s['damp'] = damp
        s.pop('method',None)
        s.pop('init',None)

        for c in sensor_configs:
            s['use_L'], s['use_OD'],s['use_CP'],s['use_LP'] = c
            scenarios.append(s.copy())

    if outfile is not None:
        size = len(scenarios)
        chunks = chunk(scenarios, size/N)
        for i,chunkk in enumerate(chunks):
            dump(chunkk,outfile % (damp,i))

def test_basic(iterations=1,proportions=[1],solvers=['LS'],
               nrow_min=3,nrow_max=4,row_step=1,ncol_min=4,ncol_max=5,col_step=1,
               EQ_NLP_max=122,EQ_NB_max=128,EQ_NS_max=128,EQ_NL_max=128,
               init=False,sparse=False):
    scenarios = []

    for i in xrange(iterations):
        s = new_s()
        s['trial'] = i
        for solver in solvers:
            s['solver'], s['sparse'] = solver, sparse
            if solver == 'LS':
                # TODO use the results of test_least_squares here
                s['method'], s['init'] = 'BB', init

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

def test_UE_basic(models=['UE','SO'],iterations=1,proportions=[1],solvers=['LS'],
            nrow_min=3,nrow_max=4,row_step=1,ncol_min=4,ncol_max=5,col_step=1,
            EQ_NLP_max=122,EQ_NB_max=128,EQ_NS_max=128,EQ_NL_max=128):
    scenarios = []

    for i in xrange(iterations):
        s = new_s()
        s['trial'] = i
        for solver in solvers:
            s['solver'], s['sparse'] = solver, False
            if solver == 'LS':
                # TODO use the results of test_least_squares here
                s['method'], s['init'] = 'BB', False

            for model in models:
                s['model'] = model
                # print json.dumps(s, sort_keys=True, indent=4 * ' ')
                # P matrices, scaling cellpath and linkpath sensing
                if solver == 'BI' or i >= 1:
                    continue
                for prop in proportions:
                    s = set_sensors(s, prop, EQ_NLP_max, EQ_NB_max,
                                    EQ_NS_max, EQ_NL_max)
                    s['sparse'] = True if model == 'SO' else False
                    scenarios.append(s.copy())
    return scenarios

def test_UE(outfile='scenarios_UESO.txt',models=['UE','SO']):
    scenarios = []
    proportions = [0.1, 0.18, 0.25, 0.33, 0.5, 0.66, 0.75, 0.9, 0.95, 1]
    sensors = [(122,128,128,128),(122,0,0,0),(0,128,0,0),(0,0,128,0),(0,0,0,128),
        (122,128,0,0),(122,0,128,0),(0,128,128,0),(0,0,128,128),(0,128,0,128),
        (122,128,128,0)]

    for EQ_NLP_max, EQ_NB_max, EQ_NS_max, EQ_NL_max in sensors:
        s = test_UE_basic(models=models,proportions=proportions,
                  EQ_NLP_max=EQ_NLP_max,EQ_NB_max=EQ_NB_max,EQ_NS_max=EQ_NS_max,
                  EQ_NL_max=EQ_NL_max)
        scenarios.extend(s)

    for s in scenarios:
        check_scenario(s)

    if outfile is not None:
        dump(scenarios,outfile)

    return scenarios


def test_all(CS_only=False,outfile='scenarios_all.txt',init=False,sparse=False):
    """
    Mega test of most things

    Note: we have to test CS separately due to its dependence on MATLAB
    :param CS_only:
    :param outfile:
    :return:
    """
    iterations = 100
    solvers = ['CS'] if CS_only else ['LS']
    proportions = [0.1, 0.18, 0.25, 0.33, 0.5, 0.66, 0.75, 0.9, 0.95, 1]
    EQ_NLP_max, EQ_NB_max, EQ_NS_max, EQ_NL_max = 122, 128, 128, 128
    nrow_min, nrow_max, ncol_min, ncol_max, row_step, col_step = 1,11,2,11,3,2

    scenarios = test_basic(iterations=iterations,proportions=proportions,
                           solvers=solvers,nrow_min=nrow_min,nrow_max=nrow_max,
                           row_step=row_step,ncol_min=ncol_min,
                           ncol_max=ncol_max,col_step=col_step,
                           EQ_NLP_max=EQ_NLP_max,EQ_NB_max=EQ_NB_max,
                           EQ_NS_max=EQ_NB_max,EQ_NL_max=EQ_NL_max,
                           init=init,sparse=sparse)

    for s in scenarios:
        check_scenario(s)

    if init:
        outfile = '%s.init' % outfile
    if sparse:
        outfile = '%s.sparse' % outfile
    if outfile is not None:
        dump(scenarios,outfile)

    return scenarios

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

def test_test(outfile='scenarios_test.txt'):
    """
    For testing that EC2 instance setup is working
    :return:
    """
    scenarios = []

    scenarios.extend(random.sample(test_all(outfile=None),5))
    scenarios.extend(test_all_links(outfile=None, n=3))
    scenarios.extend(test_least_squares(outfile=None,n=10))
    scenarios.extend(test_bayesian_inference(outfile=None,n=3))

    for s in scenarios:
        check_scenario(s)

    if outfile is not None:
        dump(scenarios,outfile)

    return scenarios


if __name__ == "__main__":
    # test_test()
    test_bayesian_inference()
    # test_small()
    # test_all()
    # test_least_squares(n=1000)
    # test_all_links(n=1000)
    # test_all_sampled()
    # test_UE()
    # test_all_links(outfile='scenarios_all_links_LBFGS.txt',method='LBFGS')
    # test_all_links(outfile='scenarios_all_links_DORE.txt',method='DORE')
    # test_all(init=True)
    # test_all(init=True,sparse=True)
    # test_all(sparse=True)
    # test_all_reduced_sensors()
    # test_all_links_UE(v=2)
    # test_all_reduced_reduced_sensors()
    # test_LSQR()
