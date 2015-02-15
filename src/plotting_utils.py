from __future__ import division
import ipdb
import json
import os

from pylab import connect, scatter
from AnnoteFinder import AnnoteFinder
import numpy as np

import config as c
from synthetic_traffic.synth_utils import to_np


def filter_valid(d):
    if 'error' in d:
        return False
    if 'duration' not in d:
        return False
    return True


def filter_v2(d):
    if 'nLinks' not in d and 'nOD' not in d and 'nCP' not in d \
            and 'nLP' not in d:
        return False
    return True


def filter_v3(d):
    if 'use_L' not in d['params']:
        return False
    return True


def isp(d, name=None, match=None):
    if name in d['params'] and d['params'][name] == match:
        return True
    return False


def filter_all_links(d):
    p = d['params']
    if 'all_links' in p and p['all_links'] and p['NLP'] == 0 and \
                    p['NL'] == 0 and p['NB'] == 0 and p['NS'] == 0:
        return True
    return False


def plot_scatter(x, y, c=None, s=None, label=None, info=None, alpha=1.0,
                 marker='o', vmin=None, vmax=None, legend=True):
    # x, y, c, s = rand(4, 100)
    if legend:
        scatter(x, y, 100*s, c, alpha=alpha, marker=marker, vmin=vmin,
                vmax=vmax)
    else:
        scatter(x, y, 100*s, c, alpha=alpha, marker=marker, vmin=vmin,
                vmax=vmax, label='_nolegend_')
    #fig.savefig('pscoll.eps')
    if label is not None:
        af = AnnoteFinder(x, y, label, info=info)
        connect('button_press_event', af)


def filter(s, group_by=(), match_by=(), geq=(), leq=()):
    d = {} if group_by is not None else []
    valid = ['nroutes','nsensors','blocks','percent flow allocated incorrectly',
             'NLPCP','use_L','use_OD','use_CP','use_LP','duration',
             'blocks_to_routes']
    for x in s:
        match = True
        for (param, value) in match_by:
            if param in valid and get_key(x, param) != value or \
                            param in x['params'] and get_key(x, param) != value:
                match = False
                break
        if not match:
            continue
        for (param, value) in geq:
            try:
                if param in valid and get_key(x, param) < value or \
                            param in x['params'] and get_key(x, param) < value:
                    match = False
                    break
            except:
                ipdb.set_trace()
        if not match:
            continue
        for (param, value) in leq:
            if param in valid and get_key(x, param) > value or \
                    param in x['params'] and get_key(x, param) > value:
                match = False
                break
        if not match:
            continue
        try:
            if group_by is not None:
                key = frozenset([(group, get_key(x, group)) for group in \
                                 group_by])
                if key in d:
                    d[key].append(x)
                else:
                    d[key] = [x]
            else:
                d.append(x)
        except TypeError:
            ipdb.set_trace()
    return d


def get_key(d, key):
    if key == 'nroutes':
        try:
            return d['AA'][1]
        except TypeError:
            ipdb.set_trace()
    elif key == 'nobj':
        return d['AA'][0]
    elif key in ['perflow', 'percent flow allocated incorrectly']:
        return _get_per_flow(d)
    elif key == 'blocks':
        return _get_blocks(d)
    elif key == 'blocks_to_routes':
        return get_key(d, 'blocks') / get_key(d, 'nroutes')
    elif key == 'max_links':
        return _get_max_links(d)
    ## Sensor configuration/constraints retrieval
    # CP/LP sensor configuration
    elif key == 'NCP':
        NCP = d['params']['NB'] + d['params']['NS'] + d['params']['NL']
        return get_key(d, 'use_CP') * NCP
    elif key == 'NLP':
        return get_key(d, 'use_LP') * d['params']['NLP']
    elif key == 'NLPCP':
        return get_key(d, 'NCP') + get_key(d, 'NLP')
    # Links/OD configurations/constraints
    elif key == 'nLinks':
        # WARNING: v2+
        return get_key(d, 'use_L') * d[key] if key in d else 0
    elif key == 'nOD':
        # WARNING: v2+
        return get_key(d, 'use_OD') * d[key] if key in d else 0
    elif key == 'nLiOD':
        return get_key(d, 'nLinks') + get_key(d, 'nOD')
    # CP/LP constraints
    elif key == 'nCP':
        # WARNING: v2+
        return get_key(d, 'use_CP') * d[key] if key in d else 0
    elif key == 'nLP':
        # WARNING: v2+
        return get_key(d, 'use_LP') * d[key] if key in d else 0
    elif key == 'nLPCP':
        return get_key(d, 'nLP') + get_key(d, 'nCP')
    # Total sensors
    elif key == 'nsensors':
        return get_key(d, 'nLinks') + get_key(d, 'nOD') + get_key(d, 'NLPCP')
    # Total constraints
    elif key == 'nconstraints':
        return get_key(d, 'nLinks') + get_key(d, 'nOD') + get_key(d, 'nLPCP')
    elif key in ['duration']:
        return d[key] if key in d else 0
    elif key in ['use_L', 'use_OD', 'use_CP', 'use_LP', 'model', 'solver',
                 'nrow', 'ncol', 'init', 'sparse', 'nodroutes', 'method']:
        # WARNING: v2+
        # Default for sensor toggle is True
        return d['params'][key] if key in d['params'] else True
    else:
        try:
            return d['params'][key]
        except:
            ipdb.set_trace()


def _get_max_links(d):
    p = d['params']
    if 'nrow' in p and 'ncol' in p:
        return 2 * (p['nrow'] * p['ncol'] * 2 + p['ncol'] + p['nrow'] - 2)
    elif p['model'] in ['UE', 'SO']:
        return 122
    print d
    return None


def _get_blocks(d):
    if d['blocks'] is None:
        return 0
    if type(d['blocks']) == int:
        return d['blocks']
    return d['blocks'][0]


def _get_per_flow(d):
    if type(d['percent flow allocated incorrectly']) == type([]):
        return d['percent flow allocated incorrectly'][-1]
    else:
        return d['percent flow allocated incorrectly']


def get_stats(xs, f, stat='mean'):
    return np.array([_get_stat(x, f, stat=stat) for x in xs])


def _get_stat(l, f, stat='first'):
    if type(l) == type({}):
        return f(l)
    if stat == 'mean':
        return np.mean([f(x) for x in l])
    elif stat == 'max':
        return np.max([f(x) for x in l])
    elif stat == 'min':
        return np.min([f(x) for x in l])
    elif stat == 'median':
        return np.median([f(x) for x in l])
    elif stat == 'all':
        return [f(x) for x in l]
    elif stat == 'first':
        return f(l[0])
    elif stat == 'last':
        return f(l[-1])
    else:
        print "Error: stat %s not found" % stat


def get_step(d, error=0.01, param='cum_times'):
    if d['params']['solver'] == 'BI':
        return d['duration']
    for i in range(len(d['percent flow allocated incorrectly'])):
        if d['percent flow allocated incorrectly'][i] <= error:
            if param == 'cum_times':
                return sum(d['times'][0:i+1])
    return d['duration']


def load_output(no_lsqr=False):
    """
    Load results from old format (run on AWS)
    CAUTION: these results aren't reproducible
    :return:
    """
    files = os.listdir(c.RESULT_DIR)
    # files = ['output_Experiment.txt']

    scenarios = []
    scenarios_all_links = []
    scenarios_v2 = []
    scenarios_v3 = []

    for f in files:
        filename = "%s/%s" % (c.RESULT_DIR, f)
        if os.path.isdir(filename):
            continue
        print '.',
        with open(filename) as out:
            for line in out:
                if line[0] != '{':
                    continue

                # Exclude invalid json strings
                try:
                    d = json.loads(line)
                except ValueError:
                    # ipdb.set_trace()
                    continue

                # Exclude invalid dicts
                if not filter_valid(d):
                    continue
                try:
                    # Exclude LSQR results
                    if no_lsqr and d['params']['solver'] == 'LSQR':
                        continue
                except TypeError:
                    ipdb.set_trace()

                # Minor correction: get rid of 'var'
                if 'var' in d:
                    d['var'] = to_np(d['var'])
                # Minor correction: for <v2, default NL/NB/NS=100
                if not filter_v3(d):  # FIXME add new versions as needed
                    if d['params']['NL'] == 0:
                        d['params']['NL'] = 100
                    if d['params']['NB'] == 0:
                        d['params']['NB'] = 100
                    if d['params']['NS'] == 0:
                        d['params']['NS'] = 100

                if d['params']['solver'] in ['BI', 'LSQR']:
                    if d['duration'] in [0, None]:
                        d['duration'] = d['init_time']

                # Exclude trivial cases
                if get_key(d, 'nroutes') == get_key(d, 'blocks'):
                    continue

                scenarios.append(d)
                if filter_all_links(d):
                    scenarios_all_links.append(d)
                if filter_v2(d):
                    scenarios_v2.append(d)
                if filter_v3(d):
                    scenarios_v3.append(d)

    return scenarios, scenarios_all_links, scenarios_v2, scenarios_v3
