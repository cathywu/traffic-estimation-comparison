import ipdb
import os
import json
from random import shuffle

from scenarios_plot import get_key, load_results, filter
from generate_scenarios import new_s, dump, check_scenario

def chunk(l, n):
    """ Yield successive n-sized chunks from l.
    """
    for i in xrange(0, len(l), n):
        yield l[i:i+n]

def test_remainders(scenario_files, result_files):
    # Tally up scenario files
    scenarios = {}
    for file in scenario_files:
        filepath = "%s/%s" % (c.SCENARIO_DIR,file)
        if os.path.isdir(filepath):
            continue
        print file
        with open(filepath) as f:
            for line in f:
                d = json.loads(line)
                d['trial'] = 0
                if d['solver'] == 'BI':
                    continue
                key = frozenset(d.iteritems())
                if key not in scenarios:
                    scenarios[key] = 1
                else:
                    scenarios[key] += 1

    # Tally up results files
    results = {}
    for file in result_files:
        filepath = "%s/%s" % (c.RESULT_DIR,file)
        if os.path.isdir(filepath):
            continue
        print file
        with open(filepath) as f:
            for line in f:
                if line[0] != '{':
                    continue

                try:
                    d = json.loads(line)
                except ValueError:
                    # ipdb.set_trace()
                    continue
                p = d['params']
                p['trial'] = 0
                key = frozenset(p.iteritems())
                if key not in results:
                    results[key] = 1
                else:
                    results[key] += 1

    # Generate the remainder
    num_files = 10
    size = len(scenarios)
    chunks = chunk(scenarios.items(), size/num_files)
    for i,chunkk in enumerate(chunks):
        with open('%s/%s.%s' % (c.SCENARIO_DIR,remainder,i), 'w') as f:
            for (k,v) in chunkk:
                if k not in results:
                    count = v
                else:
                    count = max(0,v-results[k])
                for i in xrange(count):
                    d = dict([x for x in k])

                    f.write("%s\n" % json.dumps(d))

def test_from_good_LSQR(result_files,outfile='scenarios_fromLSQR_%s.%d.%d.txt',
                        solver='LS',N=1,suffix=0):
    # Tally up scenario files
    scenarios = []

    scenarios_all, scenarios_all_links, scenarios_v2, scenarios_v3 = load_results()
    match_by = [('solver','LSQR')]
    leq = [('percent flow allocated incorrectly', 0.5)]
    geq = [('percent flow allocated incorrectly', 1e-5)]
    scenarios_LSQR = filter(scenarios_all,group_by=None,match_by=match_by,
                            leq=leq,geq=geq)
    scenarios_BI = filter(scenarios_all,group_by=None,match_by=[('solver','BI')])
    scenarios_LS = filter(scenarios_all,group_by=None,match_by=[('solver','LS')])
    scenarios_CS = filter(scenarios_all,group_by=None,match_by=[('solver','CS')])
    d_LS, d_CS, d_BI = {}, {}, {}
    for s in scenarios_LS:
        d_LS[frozenset(s['params'].iteritems())] = 1
    for s in scenarios_CS:
        d_CS[frozenset(s['params'].iteritems())] = 1
    for s in scenarios_BI:
        d_BI[frozenset(s['params'].iteritems())] = 1

    for x in scenarios_LSQR:
        s = new_s(s=x['params'])
        if s['use_CP'] == False and s['use_OD'] == False:
            continue
        else:
            types = int(s['use_L'])+int(s['use_OD'])+int(s['use_CP'])+int(s['use_LP'])
            if types < 2:
                continue
        if solver == 'LS':
            s['solver'] = solver
            s['method'] = 'BB'
            s['init'] = True
            if frozenset(s.iteritems()) not in d_LS:
                scenarios.append(s.copy())
            s['init'] = False
            if frozenset(s.iteritems()) not in d_LS:
                scenarios.append(s.copy())
        if solver in ['BI','CS']:
            s['solver'] = solver
            if (solver == 'BI' and frozenset(s.iteritems()) not in d_BI) or \
                    (solver == 'CS' and frozenset(s.iteritems()) not in d_CS):
                scenarios.append(s.copy())

    for s in scenarios:
        check_scenario(s)

    shuffle(scenarios)

    if outfile is not None:
        size = len(scenarios)
        chunks = chunk(scenarios, size/N)
        for i,chunkk in enumerate(chunks):
            dump(chunkk,outfile % (solver,suffix,i))

    return scenarios

if __name__ == "__main__":
    import config as c
    result_files = os.listdir(c.RESULT_DIR)
    scenario_files = os.listdir(c.SCENARIO_DIR)
    remainder = 'remainder/scenarios_remainder.txt'

    exclude = [ 'scenarios_test.txt', 'scenarios_temp.txt',
                'scenarios_small.txt', 'scenarios_all_sampled.txt' ]
    for f in exclude:
        if f in scenario_files:
            scenario_files.remove(f)

    # test_remainders(scenario_files, result_files)
    test_from_good_LSQR(result_files,suffix=2,solver='LS')
    test_from_good_LSQR(result_files,suffix=2,solver='BI')
    test_from_good_LSQR(result_files,suffix=2,solver='CS')