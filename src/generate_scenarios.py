import simplejson as json
import numpy as np

def set_sensors(s, prop, NLP_max, NB_max, NS_max, NL_max):
    s['NLP'] = np.ceil(NLP_max * prop)
    s['NB'] = np.ceil(NB_max * prop)
    s['NS'] = np.ceil(NS_max * prop)
    s['NL'] = np.ceil(NL_max * prop)
    return s

def test_all_links():
    """
    Test performance of solvers under the condition where all links are observed
    :return:
    """
    fname = 'scenarios_all_links.txt'
    pass

def test_least_squares():
    """
    Test performance of various methods (BB/LBFGS/DORE) of the LS solver in
    controlled manners
    :return:
    """

    # Sample UE configuration

    # Sample SO configuration

    # Sample P configuration
    fname = 'scenarios_ls.txt'
    pass

def test_noise():
    fname = 'scenarios_noise.txt'
    pass

def test_all():
    fname = 'scenarios_all.txt'
    iterations = 2
    solvers = ['CS','BI','LS']
    proportions = [0.1, 0.18, 0.25, 0.33, 0.5, 0.66, 0.75, 0.9, 0.95, 1]
    EQ_NLP_max, EQ_NB_max, EQ_NS_max, EQ_NL_max = 122, 128, 128, 128

    with open(fname,'w') as out:
        for i in xrange(iterations):
            s = {}
            s['trial'] = i
            for solver in solvers:
                s['solver'], s['sparse'] = solver, False
                if solver == 'LS':
                    # TODO use the results of test_least_squares here
                    s['method'], s['init'] = 'BB', True

                models = ['UE','SO','P']

                for model in models:
                    s['model'] = model
                    # print json.dumps(s, sort_keys=True, indent=4 * ' ')
                    # P matrices, scaling cellpath and linkpath sensing
                    if model == 'P':
                        nrows, ncols = range(1,11,3), range(2,11,2)
                        for nrow in nrows:
                            for ncol in ncols:
                                s['nrow'], s['ncol'] = nrow, ncol
                                s['nodroutes'] = 15 # FIXME more configurations?
                                NLP_max = (ncol-1) * (nrow-1) * 2 + ncol + nrow - 2
                                NB_max = ncol * nrow * 8
                                NS_max, NL_max = 0, NLP_max
                                for prop in proportions:
                                    s = set_sensors(s, prop, NLP_max, NB_max, NS_max, NL_max)
                                    for sparse in [True, False]:
                                        s['sparse'] = sparse
                                        out.write('%s\n' % json.dumps(s)) # WRITE

                    # UE/SO matrices, scaling cellpath and linkpath sensing
                    else:
                        for prop in proportions:
                            s = set_sensors(s, prop, EQ_NLP_max, EQ_NB_max,
                                            EQ_NS_max, EQ_NL_max)
                            s['sparse'] = True if model == 'SO' else False
                            out.write('%s\n' % json.dumps(s)) # WRITE

def test_small():
    """
    For testing that hadoop setup is working
    :return:
    """
    fname = 'scenarios_small.txt'
    iterations = 2
    solvers = ['LS']
    proportions = [0.5, 1]
    EQ_NLP_max, EQ_NB_max, EQ_NS_max, EQ_NL_max = 122, 128, 128, 128

    with open(fname,'w') as out:
        for i in xrange(iterations):
            s = {}
            s['trial'] = i
            for solver in solvers:
                s['solver'], s['sparse'] = solver, False
                if solver == 'LS':
                    # TODO use the results of test_least_squares here
                    s['method'], s['init'] = 'BB', True

                models = ['UE','SO','P']

                for model in models:
                    s['model'] = model
                    # print json.dumps(s, sort_keys=True, indent=4 * ' ')
                    # P matrices, scaling cellpath and linkpath sensing
                    if model == 'P':
                        nrows, ncols = range(3,4), range(4,5)
                        for nrow in nrows:
                            for ncol in ncols:
                                s['nrow'], s['ncol'] = nrow, ncol
                                s['nodroutes'] = 15 # FIXME more configurations?
                                NLP_max = (ncol-1) * (nrow-1) * 2 + ncol + nrow - 2
                                NB_max = ncol * nrow * 8
                                NS_max, NL_max = 0, NLP_max
                                for prop in proportions:
                                    s = set_sensors(s, prop, NLP_max, NB_max, NS_max, NL_max)
                                    for sparse in [True, False]:
                                        s['sparse'] = sparse
                                        out.write('%s\n' % json.dumps(s)) # WRITE

                    # UE/SO matrices, scaling cellpath and linkpath sensing
                    else:
                        for prop in proportions:
                            s = set_sensors(s, prop, EQ_NLP_max, EQ_NB_max,
                                            EQ_NS_max, EQ_NL_max)
                            s['sparse'] = True if model == 'SO' else False
                            out.write('%s\n' % json.dumps(s)) # WRITE

if __name__ == "__main__":
    test_all()
    test_small()
    # test_all_links()
    # test_least_squares()
