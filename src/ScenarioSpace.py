from __future__ import division
import ipdb
import os
import json
from pprint import pprint

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, show

import config as c
from scenario_utils import load, new_s, NumpyAwareJSONEncoder, args_from_SC, \
    args_from_solver, args_from_TN
from plotting_utils import plot_scatter, filter, get_stats, get_key, get_step, \
    load_output
from generate_scenarios import dump


class ScenarioSpace:
    def __init__(self, no_lsqr=False):
        # Plot configuration
        self.init = False
        self.sparse = False
        self.yaxis = 'blocks'
        self.error = 0.10
        self.error2 = 0.30
        self.error3 = 0.50

        self.no_lsqr = no_lsqr

    def get_args(self, s):
        """
        Generates args dictionary based on Scenario object
        :param s:
        :return:
        """
        args = new_s()
        # TrafficNetwork args
        args = args_from_TN(s.TN, args)
        # Sensor configuration args
        args = args_from_SC(s.SC, args)
        # Solver args
        args = args_from_solver(s.solver, args)
        return args

    def scenarios_to_output(self):
        """
        Takes the pickled Scenario objects in c.RESULTS_DIR and converts them
        to the output format compatible with the plotting code
        :return:
        """
        output_fname = "%s/output_Experiment.txt" % c.RESULT_DIR
        dirs = [c.SCENARIO_DIR_NEW, '%s/temp' % c.SCENARIO_DIR_NEW]
        with open(output_fname, 'w') as out:
            for dir in dirs:
                files = os.listdir(dir)
                for f in files:
                    fname = "%s/%s" % (dir, f)
                    try:
                        s = load(fname)
                        params = self.get_args(s)
                        output = s.output
                        output['params'] = params
                        out.write('%s\n' % json.dumps(output,
                                                      cls=NumpyAwareJSONEncoder))
                        # print 'Loaded fine: %s' % fname
                    except (EOFError, AttributeError):
                        print 'rm %s' % fname
                        pass

    def load_output(self):
        """
        Load results from old format (run on AWS)
        CAUTION: these results aren't reproducible
        :return:
        """
        self.scenarios, self.scenarios_all_links, self.scenarios_v2, \
        self.scenarios_v3 = load_output(no_lsqr=self.no_lsqr)

    def load_scenarios(self):
        """
        Load results from new format (via Experiment.py)
        :return:
        """
        pass

    def generate_statistics(self, error=1, model='P'):
        print
        print "Number of (valid) scenarios: %s" % len(self.scenarios)

        match_by = [('model', model)]

        leq=[('percent flow allocated incorrectly', error)]
        d = filter(self.scenarios, match_by=match_by, leq=leq)
        print "Number with error <=%s: %s" % (error,len(d.values()[0]))

        group_by = ['sparse']
        d = filter(self.scenarios, match_by=match_by, group_by=group_by, leq=leq)
        print "Of those we have",
        pprint([(dict(k),len(v)) for k,v in d.iteritems()])

        group_by.append('solver')
        d = filter(self.scenarios, match_by=match_by, group_by=group_by, leq=leq)
        print "Of those we have",
        pprint([(dict(k),len(v)) for k,v in d.iteritems()])

        group_by = ['NCP', 'NLP', 'sparse']
        if model == 'P':
            group_by.extend(['nrow', 'ncol'])
        d = filter(self.scenarios, match_by=match_by, group_by=group_by, leq=leq)
        ntypes = len(d.keys())
        print 'Number of comparison types: %s' % ntypes
        undones = []
        NCPs, NLPs = [], []
        for (k,v) in d.iteritems():
            params = [vv['params'] for vv in v]
            undone = ScenarioSpace.generate_missing_solver_params(params)
            undones.extend(undone)
            dd = filter(v, group_by=['solver'])
            solvers = get_stats(dd.itervalues(), lambda x: get_key(x, 'solver'),
                                stat='first')
            NCPs.append(dict(k)['NCP'])
            NLPs.append(dict(k)['NLP'])
            if len(undone) > 0 and 'CS' not in set(solvers):
                print dict(k), set(solvers)
                # pprint(undone)
        print 'NCPs', set(NCPs)
        print 'NLPs', set(NLPs)

        dump(undones, 'scenarios_comparison.txt')


    @staticmethod
    def generate_missing_solver_params(params):
        undone = []

        intify = ['NLP']
        for p in params:
            p['NB'] = get_key({'params': p}, 'NB') + \
                      get_key({'params': p}, 'NL') + \
                      get_key({'params': p}, 'NS')
            p['NL'], p['NS'] = 0, 0
            for pp in intify:
                p[pp] = get_key({'params': p}, pp)
            if 'trial' in p:
                del p['trial']
            if 'all_links' in p and p['model'] == 'P':
                del p['all_links']
            for k in ['use_L', 'use_CP', 'use_LP', 'use_OD']:
                if k in p:
                    del p[k]
        fset = [frozenset(new_s(p).iteritems()) for p in params]

        template = params[0].copy()
        mask = ['method','init','sparse_BI']
        for m in mask:
            if m in template:
                del template[m]
        template = new_s(s=template)

        solver_configs = [(('solver','LS'), ('method','BB'), ('init',True)),
                          (('solver','LS'), ('method','BB'), ('init',False)),
                          (('solver','BI'), ('sparse_BI',True)),
                          (('solver','BI'), ('sparse_BI',False)),
                          (('solver','CS'), ('method','cvx_oracle')),
                          (('solver','CS'), ('method',
                                    'cvx_random_sampling_L1_6000_replace')),
                          ]
        for sc in solver_configs:
            done = False
            temp = template.copy()
            for (k,v) in sc:
                temp[k] = v
            ftemp = frozenset(temp.iteritems())
            for f in fset:
                if ftemp == f:
                    done = True
                    # print '[Done] %s' % temp
                    break
            if not done:
                undone.append(temp)
                # print '[Need to do]: %s' % temp
        return undone

    def plot_size_vs_speed(self, stat='mean',
                           caption=None, error_leq=0.01, max_NLPCP=None,
                           model=None, disp=True, ylim=None, b2n_geq=0,
                           b2n_leq=1, solvers = ('LS', 'BI', 'LSQR', 'CS'),
                           color_axis='b2n'):


        def plot1(s, title, stat='mean', marker='o', error_leq=0.01, ylim=None,
                  colorbar=True, color_axis='b2n'):
            nroutes = get_stats(s, lambda x: get_key(x, 'nroutes'), stat=stat)
            duration = get_stats(s, lambda x: get_key(x, 'duration'), stat=stat)
            blocks_to_routes = get_stats(s,
                                         lambda x: get_key(x,
                                                           'blocks_to_routes'),
                                         stat=stat)
            duration_leq_error = [get_step(x, error=error_leq,
                                           param='cum_times') for x in s]

            blocks = get_stats(s, lambda x: get_key(x, 'blocks'), stat=stat)
            NLP = get_stats(s, lambda x: get_key(x, 'NLP'), stat=stat)
            NCP = get_stats(s, lambda x: get_key(x, 'NCP'), stat=stat)
            nLPconstraints = get_stats(s, lambda x: get_key(x, 'nLP'),
                                       stat=stat)
            nCPconstraints = get_stats(s, lambda x: get_key(x, 'nCP'),
                                       stat=stat)

            perflow_wrong = get_stats(s, lambda x: get_key(x, 'perflow'),
                                      stat=stat)

            note = [{'nLP constraints': a, 'nCP constraints': b, 'NLP': k,
                     'NCP': e, 'blocks': f, 'duration': "{:.5f}".format(g),
                     'perflow wrong': "{:.5f}".format(h),
                    } for (a, b, k, e, f, g, h) in zip(nLPconstraints,
                                                       nCPconstraints,
                                                       NLP, NCP, blocks,
                                                       duration,
                                                       perflow_wrong)]
            info = s

            if color_axis == 'error':
                colors = perflow_wrong
            elif color_axis == 'b2n':
                colors = blocks_to_routes
            else:
                return NotImplemented
            size = 2
            labels = [json.dumps(x, sort_keys=True, indent=4) for x in note]

            plot_scatter(nroutes, duration_leq_error, c=colors, s=size,
                         label=labels, info=info, alpha=0.2, marker=marker)
            plt.hold(True)

            if colorbar:
                cb = plt.colorbar()
                cb.set_alpha(1)
                cb.draw_all()
                if color_axis == 'error':
                    cb.set_label('% route flow error')
                elif color_axis == 'b2n':
                    cb.set_label('Blocks to routes')
                else:
                    return NotImplemented

            plt.title(title)
            plt.xlabel('Number of routes')
            plt.ylabel('Computation time (sec)')
            if ylim is not None:
                plt.ylim(ylim[0], ylim[1])
            else:
                plt.ylim(np.max([plt.ylim()[0]], 0), plt.ylim()[1])
            plt.xlim(np.max([0, plt.xlim()[0]]), plt.xlim()[1])

        suptitle = "Size = fixed [solver=%s,model=%s,stat=%s,LP=^,CP=v]"
        title1 = 'Size vs speed'

        markers = ['*', 'd', '+', 'x']
        leq = [('percent flow allocated incorrectly', error_leq),
               ('blocks_to_routes', b2n_leq)]
        # filter for tests that took some time (not just initial soln)
        geq = [('duration', 1e-8), ('blocks_to_routes', b2n_geq)]
        fig = plt.figure()
        print
        for solver, m in zip(solvers, markers):
            match_by = [('model', 'P'), ('solver', solver)]
            if max_NLPCP is not None:
                leq.append(('NLPCP', max_NLPCP))

            d = filter(self.scenarios_v2, match_by=match_by, leq=leq, geq=geq)
            if len(d.keys()) > 0:
                s = d.values()[0]
                print '[%s (%s): %s]' % (solver, m, len(s)),
                fig.suptitle("%s %s" % (suptitle % (solver, model,
                                                    stat), caption), fontsize=8)
                colorbar = True if m == '*' else False
                plot1(s, title1, stat=stat, marker=m, error_leq=error_leq,
                      colorbar=colorbar, ylim=ylim, color_axis=color_axis)
                plt.hold(True)
        if disp:
            show()

    def plot_sparsity_vs_error(self, solvers=('LS', 'BI', 'LSQR', 'CS'),
                               init=False, sparse=True, stat='mean',
                               caption=None, error_leq=0.01, max_NLPCP=None,
                               model=None, damp=0.0, disp=True, ylim=None,
                               xlim=None, color_axis='b2n'):

        def plot1(s, title, stat='mean', marker='o', ylim=None, colorbar=True,
                  color_axis='error'):
            nroutes = get_stats(s, lambda x: get_key(x, 'nroutes'), stat=stat)
            blocks_to_routes = get_stats(s, lambda x: get_key(x,
                                                            'blocks_to_routes'),
                                         stat=stat)
            nrows = get_stats(s, lambda x: get_key(x, 'nrow'), stat=stat)
            ncols = get_stats(s, lambda x: get_key(x, 'ncol'), stat=stat)
            sparsity = [nrow * ncol * 2 / nroute for (nrow, ncol, nroute) in
                        zip(nrows, ncols, nroutes)]
            duration = get_stats(s, lambda x: get_key(x, 'duration'), stat=stat)

            blocks = get_stats(s, lambda x: get_key(x, 'blocks'), stat=stat)
            NLP = get_stats(s, lambda x: get_key(x, 'NLP'), stat=stat)
            NCP = get_stats(s, lambda x: get_key(x, 'NCP'), stat=stat)
            nLPconstraints = get_stats(s, lambda x: get_key(x, 'nLP'),
                                       stat=stat)
            nCPconstraints = get_stats(s, lambda x: get_key(x, 'nCP'),
                                       stat=stat)

            perflow_wrong = get_stats(s, lambda x: get_key(x, 'perflow'),
                                      stat='min')

            note = [{'nLP constraints': a, 'nCP constraints': b, 'NLP': k,
                     'NCP': e, 'blocks': f, 'duration': "{:.5f}".format(g),
                     'perflow wrong': "{:.5f}".format(h),
                    } for (a, b, k, e, f, g, h) in zip(nLPconstraints,
                                                       nCPconstraints,
                                                       NLP, NCP, blocks,
                                                       duration,
                                                       perflow_wrong)]
            info = s

            if color_axis == 'error':
                colors = perflow_wrong
            elif color_axis == 'b2n':
                colors = blocks_to_routes
            else:
                return NotImplemented
            size = 2
            labels = [json.dumps(x, sort_keys=True, indent=4) for x in note]

            plot_scatter(sparsity, perflow_wrong, c=colors, s=size,
                         label=labels, info=info, alpha=0.2, marker=marker)
            plt.hold(True)

            if colorbar:
                cb = plt.colorbar()
                cb.set_alpha(1)
                cb.draw_all()
                if color_axis == 'error':
                    cb.set_label('% route flow error')
                elif color_axis == 'b2n':
                    cb.set_label('Blocks / nroutes')
                else:
                    return NotImplemented

            plt.title(title)
            plt.xlabel('Sparsity (percent)')
            plt.ylabel('Route flow error')
            if ylim is not None:
                plt.ylim(ylim[0], ylim[1])
            else:
                plt.ylim(np.max([plt.ylim()[0]], 0), plt.ylim()[1])
            if xlim is not None:
                plt.xlim(xlim[0], xlim[1])
            else:
                plt.xlim(np.max([0, plt.xlim()[0]]), plt.xlim()[1])

        suptitle = "Size = fixed [solver=%s,model=%s,sparse=%s,init=%s,stat=%s,LP=^,CP=v]"
        title1 = 'Sparsity vs route flow error'

        markers = ['*', 'd', '+', 'x']
        leq = [('percent flow allocated incorrectly', error_leq)]
        geq = [('duration', 1e-8)]  # took some time (not just initial soln)
        fig = plt.figure()
        print
        for solver, m in zip(solvers, markers):
            match_by = [('model', 'P'), ('solver', solver)]
            if max_NLPCP is not None:
                leq.append(('NLPCP', max_NLPCP))

            d = filter(self.scenarios, match_by=match_by, leq=leq, geq=geq)
            if len(d.keys()) > 0:
                s = d.values()[0]
                print '[%s (%s): %s]' % (solver, m, len(s)),
                fig.suptitle("%s %s" % (suptitle % (solver, model, sparse, init,
                                                    stat), caption), fontsize=8)
                colorbar = True if m == '*' else False
                plot1(s, title1, stat=stat, marker=m, colorbar=colorbar,
                      ylim=ylim, color_axis=color_axis)
                plt.hold(True)
        if disp:
            plt.hold(False)
            show()

    def plot_solver_comparison(self, sparse=True, stat='mean',
                               caption=None, error_leq=1.0, error_leq2=None,
                               error_leq3=None, max_NLPCP=None, model='P',
                               disp=True, use_L=None, use_OD=None,
                               use_CP=None, use_LP=None, b2n_geq=0,
                               b2n_leq=1):

        def plot1(d, title, color_map=None, stat='mean', error_leq=1.0,
                  error_leq2=None,
                  error_leq3=None):
            # plot nroutes vs nconstraints needed to achieve accuracy
            # color: sensor config?
            # size: actual # of sensors (including ODs)
            import random

            solver_sizes = [2,3] # [1,2,3]
            marker_sizes = [1,4] # [0.25,2,6]
            for (size,msize) in zip(solver_sizes,marker_sizes):
                nrows, ncols, colors, labels, infos, markers = [], [], [], [], [], []
                for (k, v) in d.iteritems():
                    dd = filter(v, group_by=['solver'])
                    solvers = get_stats(dd.itervalues(),
                                        lambda x: get_key(x, 'solver'),
                                        stat='first')
                    perflow_wrong = get_stats(dd.itervalues(),
                                              lambda x: get_key(x, 'perflow'),
                                              stat='min')
                    # if 'BI' in solvers:
                    #     print perflow_wrong, solvers, dd.keys(), len(dd.values())
                    if solvers.size == size:
                        if solvers.size == 2:
                            print '[%s=%s], %s = %s | %s' % (solvers.size,
                                repr(solvers), len(v), repr([len(ddd) for ddd in dd.values()]),
                                repr(k))
                        # if dict(k)['nrow']*dict(k)['ncol'] <= 40:
                        #     print 'Need to test for BI: NLP=%s NCP=%s (size = (%s,%s))' % \
                        #           (dict(k)['NLP'], dict(k)['NCP'], dict(k)['nrow'],
                        #            dict(k)['ncol'])
                        best_perflow, best_solver = \
                            min(zip(perflow_wrong, solvers), key=lambda x: x[0])
                        if best_perflow < error_leq:
                            marker = 'o'
                        elif error_leq2 is not None and best_perflow < error_leq2:
                            marker = '^'
                        elif error_leq3 is not None and best_perflow < error_leq3:
                            marker = 'v'
                        else:
                            continue
                        nrow = get_stats(dd.itervalues(),
                                         lambda x: get_key(x, 'nrow'), stat=stat)[0]
                        ncol = get_stats(dd.itervalues(),
                                         lambda x: get_key(x, 'ncol'), stat=stat)[0]

                        nroutes = get_stats(dd.itervalues(),
                                            lambda x: get_key(x, 'nroutes'),
                                            stat=stat)
                        sparse = get_stats(dd.itervalues(),
                                           lambda x: get_key(x, 'sparse'),
                                           stat='first')
                        blocks = get_stats(dd.itervalues(),
                                           lambda x: get_key(x, 'blocks'),
                                           stat=stat)
                        NLP = get_stats(dd.itervalues(),
                                        lambda x: get_key(x, 'NLP'), stat=stat)
                        NCP = get_stats(dd.itervalues(),
                                        lambda x: get_key(x, 'NCP'), stat=stat)
                        nLconstraints = get_stats(dd.itervalues(),
                                                  lambda x: get_key(x, 'nLinks'),
                                                  stat=stat)
                        nODconstraints = get_stats(dd.itervalues(),
                                                   lambda x: get_key(x, 'nOD'),
                                                   stat=stat)
                        nLPconstraints = get_stats(dd.itervalues(),
                                                   lambda x: get_key(x, 'nLP'),
                                                   stat=stat)
                        nCPconstraints = get_stats(dd.itervalues(),
                                                   lambda x: get_key(x, 'nCP'),
                                                   stat=stat)
                        nTotalConstraints = get_stats(dd.itervalues(),
                                                      lambda x: get_key(x,
                                                                        'nconstraints'),
                                                      stat=stat)
                        nTotalSensors = get_stats(dd.itervalues(),
                                                  lambda x: get_key(x, 'nsensors'),
                                                  stat=stat)
                        duration = get_stats(dd.itervalues(),
                                             lambda x: get_key(x, 'duration'),
                                             stat=stat)
                        note = [{'nL constraints': a, 'nOD constraints': b,
                                 'nLP constraints': c, 'nCP constraints': e,
                                 'NLP': f, 'NCP': g, 'blocks': h,
                                 'duration': "{:.5f}".format(i),
                                 'perflow wrong': "{:.5f}".format(j),
                                 'total constraints': k,
                                 'total sensors': l, 'nroutes': "%s" % (m),
                                } for (a, b, c, e, f, g, h, i, j, k, l, m) in
                                zip(nLconstraints,
                                    nODconstraints, nLPconstraints, nCPconstraints,
                                    NLP,
                                    NCP, blocks, duration, perflow_wrong,
                                    nTotalConstraints,
                                    nTotalSensors, nroutes)]
                        info = dd.values()[0]
                        # print nroutes, blocks

                        # max_size = np.max(nTotalSensors)
                        label = json.dumps(note[0], sort_keys=True, indent=4)
                        nrows.append(nrow + random.random() - 0.5)
                        ncols.append(ncol + random.random() - 0.5)
                        colors.append(color_map[best_solver])
                        labels.append(label)
                        infos.append(info)
                        markers.append(marker)
                    else:
                        # print 'Singleton: %s e=%s %s' % (solvers[0], perflow_wrong[0], repr(k))
                        pass

                nrows = np.array(nrows)
                ncols = np.array(ncols)
                colors = np.array(colors)

                # CAUTION plotting nroutes vs accuracy will be weird because
                # LSQR considers all routes, whereas the other methods consider
                # an abridged set based on the EQ constraint
                for m in ['o', '^', 'v']:
                    idx = np.array([i for (i, x) in enumerate(markers) if x == m])
                    if idx.size == 0:
                        continue
                    curr_labels = [labels[i] for i in idx]
                    curr_info = [infos[i] for i in idx]
                    plot_scatter(nrows[idx], ncols[idx], c=colors[idx], s=msize,
                                 label=curr_labels, info=curr_info,
                                 alpha=0.2, marker=m)
                plt.hold(True)

            plt.title(title)
            plt.xlabel('Number of rows in the grid network')
            plt.ylabel('Number of cols in the grid network')

        suptitle = "[model=%s,sparse=%s,stat=%s,Top=o,Mid=^,Low=v,size=fixed,use_L=%s,use_OD=%s,use_CP=%s,use_LP=%s]"
        title1 = 'Comparison of solvers'

        leq = [('blocks_to_routes', b2n_leq)]
        geq = [('blocks_to_routes', b2n_geq)]
        group_by = ['NCP', 'NLP']
        match_by = [('model', model), ('sparse', sparse)]
        # group_by.append('use_L') if use_L is None else match_by.append(
        #     ('use_L', use_L))
        # group_by.append('use_OD') if use_OD is None else match_by.append(
        #     ('use_OD', use_OD))
        # group_by.append('use_CP') if use_CP is None else match_by.append(
        #     ('use_CP', use_CP))
        # group_by.append('use_LP') if use_LP is None else match_by.append(
        #     ('use_LP', use_LP))
        if model == 'P':
            group_by.extend(['nrow', 'ncol'])

        leq.extend([('NLPCP', max_NLPCP)] if max_NLPCP is not None else [])

        # colors = ['b','g','r','c','m','y','k']
        color_map = {'LSQR': 'b', 'LS': 'g', 'CS': 'r', 'BI': 'c'}

        fig = plt.figure()
        legend = []
        d = filter(self.scenarios, group_by=group_by, match_by=match_by,
                   leq=leq, geq=geq)
        if len(d.keys()) > 0:
            plot1(d, title1, stat=stat, color_map=color_map,
                  error_leq=error_leq,
                  error_leq2=error_leq2, error_leq3=error_leq3)
        plt.legend(legend)

        ax = plt.gca()
        ax.autoscale(True)
        fig.suptitle(
            "%s %s" % (suptitle % (model, sparse, stat, use_L, use_OD, use_CP,
                                   use_LP), caption), fontsize=8)

        if disp:
            show()


if __name__ == "__main__":
    SS = ScenarioSpace(no_lsqr=True)
    SS.scenarios_to_output()
    # SS.load_output()
    # SS.generate_statistics(error=10)

    # scenario_files = os.listdir(c.SCENARIO_DIR_NEW)
    # for sf in scenario_files:
    #     fname = '%s/%s' % (c.SCENARIO_DIR_NEW, sf)
    #     try:
    #         s = load(fname)
    #         print s.output
    #     except EOFError:
    #         print 'EOFError... %s' % fname
    #         pass
