from __future__ import division
import ipdb
import json
import os
from matplotlib.pyplot import figure, show
import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt
from pylab import plot
import config as c
from synthetic_traffic.synth_utils import to_np
from plotting_utils import plot_scatter, filter, get_stats, get_key, load_output

def plot_ls(s):
    """
    {u'AA': [861, 420],
    u'blocks': [420],
    u'0.5norm(Ax-b)^2': [0.0],
    u'percent flow allocated incorrectly': [0.0],
    u'max|f * (x_init-x_true)|': 0.0,
     u'incorrect x entries': [],
     u'times': [0],
     u'iters': [0],
     u'params': {
         u'NLP': 64.0,
         u'NL': 64.0,
         u'nrow': 7,
         u'solver': u'LS',
         u'NB': 303.0,
         u'trial': 4,
         u'init': True,
         u'ncol': 6,
         u'sparse': True,
         u'nodroutes': 15,
         u'model': u'P',
         u'NS': 0.0,
         u'method': u'BB'},
     u'duration': 0,
     u'max|f * (x-x_true)|': [0.0],
     u'0.5norm(Ax*-b)^2': 0.0,
     u'0.5norm(Ax_init-b)^2': 0.0}
    :param s:
    :return:
    """
    pass

def plot_solver_comparison(s, sparse=True, stat='mean',
                           caption=None, error_leq=1.0, error_leq2=None,
                           error_leq3=None, max_NLPCP=None, model='P',
                           damp=0.0, disp=True,use_L=None,use_OD=None,use_CP=None,
                           use_LP=None):

    def plot1(d, title, color_map=None, stat='mean', error_leq=1.0, error_leq2=None,
              error_leq3=None):
        # plot nroutes vs nconstraints needed to achieve accuracy
        # color: sensor config?
        # size: actual # of sensors (including ODs)
        import random

        nrows, ncols, colors, labels, infos, markers = [], [], [], [], [], []
        for (k,v) in d.iteritems():
            dd = filter(v,group_by=['solver'])
            solvers = get_stats(dd.itervalues(), lambda x: get_key(x,'solver'), stat='first')
            perflow_wrong = get_stats(dd.itervalues(), lambda x: get_key(x,'perflow'), stat=stat)
            if solvers.size > 1:
                best_perflow, best_solver = \
                    min(zip(perflow_wrong,solvers),key=lambda x: x[0])
                if best_perflow < error_leq:
                    marker = 'o'
                elif error_leq2 is not None and best_perflow < error_leq2:
                    marker = '^'
                elif error_leq3 is not None and best_perflow < error_leq3:
                    marker = 'v'
                nrow = get_stats(dd.itervalues(), lambda x: get_key(x,'nrow'), stat=stat)[0]
                ncol = get_stats(dd.itervalues(), lambda x: get_key(x,'ncol'), stat=stat)[0]

                nroutes = get_stats(dd.itervalues(), lambda x: get_key(x,'nroutes'), stat=stat)
                sparse = get_stats(dd.itervalues(), lambda x: get_key(x,'sparse'), stat='first')
                blocks = get_stats(dd.itervalues(), lambda x: get_key(x,'blocks'), stat=stat)
                NLP = get_stats(dd.itervalues(), lambda x: get_key(x,'NLP'), stat=stat)
                NCP = get_stats(dd.itervalues(), lambda x: get_key(x,'NCP'), stat=stat)
                nLconstraints = get_stats(dd.itervalues(), lambda x: get_key(x,'nLinks'), stat=stat)
                nODconstraints = get_stats(dd.itervalues(), lambda x: get_key(x,'nOD'), stat=stat)
                nLPconstraints = get_stats(dd.itervalues(), lambda x: get_key(x,'nLP'), stat=stat)
                nCPconstraints = get_stats(dd.itervalues(), lambda x: get_key(x,'nCP'), stat=stat)
                nTotalConstraints = get_stats(dd.itervalues(), lambda x: get_key(x,'nconstraints'), stat=stat)
                nTotalSensors = get_stats(dd.itervalues(), lambda x: get_key(x,'nsensors'), stat=stat)
                duration = get_stats(dd.itervalues(), lambda x: get_key(x,'duration'), stat=stat)
                note = [{'nL constraints': a, 'nOD constraints': b,
                         'nLP constraints': c, 'nCP constraints': e, 'NLP': f, 'NCP': g,
                         'blocks' : h, 'duration' : "{:.5f}".format(i),
                         'perflow wrong' : "{:.5f}".format(j), 'total constraints': k,
                         'total sensors': l, 'nroutes':"%s" % (m),
                         } for (a,b,c,e,f,g,h,i,j,k,l,m) in zip(nLconstraints,
                            nODconstraints,nLPconstraints,nCPconstraints,NLP,
                            NCP,blocks,duration,perflow_wrong,nTotalConstraints,
                            nTotalSensors,nroutes)]
                info = dd.values()[0]

                # max_size = np.max(nTotalSensors)
                size = 2 # 40/max_size * nTotalSensors
                label = json.dumps(note[0],sort_keys=True, indent=4)
                nrows.append(nrow+random.random()-0.5)
                ncols.append(ncol+random.random()-0.5)
                colors.append(color_map[best_solver])
                labels.append(label)
                infos.append(info)
                markers.append(marker)

        nrows = np.array(nrows)
        ncols = np.array(ncols)
        colors = np.array(colors)

        # CAUTION plotting nroutes vs accuracy will be weird because
        # LSQR considers all routes, whereas the other methods consider
        # an abridged set based on the EQ constraint
        for m in ['o','^','v']:
            idx = np.array([i for (i,x) in enumerate(markers) if x==m])
            if idx.size == 0:
                continue
            curr_labels = [labels[i] for i in idx]
            curr_info = [infos[i] for i in idx]
            plot_scatter(nrows[idx],ncols[idx],c=colors[idx],s=size,
                         label=curr_labels,info=curr_info,
                         alpha=0.2,marker=m)


        plt.title(title)
        plt.xlabel('Number of routes')
        plt.ylabel('Number of total sensors')

    suptitle = "[model=%s,sparse=%s,stat=%s,Top=o,Mid=^,Low=v,size=fixed,use_L=%s,use_OD=%s,use_CP=%s,use_LP=%s]"
    title1 = 'Comparison of solvers'

    group_by = ['NCP','NLP']
    match_by = [('model',model),('sparse',sparse)]
    group_by.append('use_L') if use_L is None else match_by.append(('use_L',use_L))
    group_by.append('use_OD') if use_OD is None else match_by.append(('use_OD',use_OD))
    group_by.append('use_CP') if use_CP is None else match_by.append(('use_CP',use_CP))
    group_by.append('use_LP') if use_LP is None else match_by.append(('use_LP',use_LP))
    if model == 'P':
        group_by.extend(['nrow','ncol'])

    leq_NLPCP = [('NLPCP',max_NLPCP)] if max_NLPCP is not None else []

    # colors = ['b','g','r','c','m','y','k']
    color_map = { 'LSQR': 'b', 'LS': 'g', 'CS': 'r', 'BI': 'c' }

    fig = plt.figure()
    legend = []
    d = filter(s,group_by=group_by, match_by=match_by, leq=leq_NLPCP)
    if len(d.keys()) > 0:
        plot1(d, title1, stat=stat, color_map=color_map, error_leq=error_leq,
              error_leq2=error_leq2, error_leq3=error_leq3)
    plt.legend(legend)

    ax = plt.gca()
    ax.autoscale(True)
    fig.suptitle("%s %s" % (suptitle % (model,sparse,stat,use_L,use_OD,use_CP,
                                        use_LP),caption), fontsize=8)

    if disp:
        show()

def plot_sensors_vs_configs_v3(s, init=False, sparse=True, stat='mean',
                               caption=None, error_leq=1, error_leq2=None,
                               error_leq3=None, max_NLPCP=None, model=None,
                               solver='LS', damp=0.0, disp=True):
    def plot_lp(d, color='#99FF99', stat='max', marker='*'):
        # Plot # links line, which indicates where LP sensors would be perfect
        if d == {} or d is None:
            return

        nroutes = get_stats(d.itervalues(), lambda x: get_key(x,'nroutes'), stat=stat)
        nroutes, idx = np.unique(nroutes, return_index=True)
        links = get_stats(d.itervalues(), lambda x: get_key(x,'max_links'), stat=stat)[idx]

        note = [{} for a in nroutes]
        info = note

        size = 2
        labels = [json.dumps(x,sort_keys=True, indent=4) for x in note]

        plot_scatter(nroutes,links,c=color,s=size,label=labels,info=info,
                     alpha=1,marker=marker,legend=False)


    def plot1(d, title, color='b', config=[True,True,True,True], stat='mean',
              marker='o'):
        # plot nroutes vs nconstraints needed to achieve accuracy
        # color: sensor config?
        # size: actual # of sensors (including ODs)

        nroutes = get_stats(d.itervalues(), lambda x: get_key(x,'nroutes'), stat=stat)
        blocks = get_stats(d.itervalues(), lambda x: get_key(x,'blocks'), stat=stat)
        NLP = get_stats(d.itervalues(), lambda x: get_key(x,'NLP'), stat=stat)
        NCP = get_stats(d.itervalues(), lambda x: get_key(x,'NCP'), stat=stat)
        nLconstraints = get_stats(d.itervalues(), lambda x: get_key(x,'nLinks'), stat=stat)
        nODconstraints = get_stats(d.itervalues(), lambda x: get_key(x,'nOD'), stat=stat)
        nLPconstraints = get_stats(d.itervalues(), lambda x: get_key(x,'nLP'), stat=stat)
        nCPconstraints = get_stats(d.itervalues(), lambda x: get_key(x,'nCP'), stat=stat)
        nTotalConstraints = get_stats(d.itervalues(), lambda x: get_key(x,'nconstraints'), stat=stat)
        nTotalSensors = get_stats(d.itervalues(), lambda x: get_key(x,'nsensors'), stat=stat)

        duration = get_stats(d.itervalues(), lambda x: get_key(x,'duration'), stat=stat)
        perflow_wrong = get_stats(d.itervalues(), lambda x: get_key(x,'perflow'), stat=stat)

        note = [{'nL constraints': a, 'nOD constraints': b,
                 'nLP constraints': c, 'nCP constraints': e, 'NLP': f, 'NCP': g,
                 'blocks' : h, 'duration' : "{:.5f}".format(i),
                 'perflow wrong' : "{:.5f}".format(j), 'total constraints': k,
                 'total sensors': l,
                 } for (a,b,c,e,f,g,h,i,j,k,l) in zip(nLconstraints,nODconstraints,
                                              nLPconstraints,nCPconstraints,NLP,
                                              NCP,blocks,duration,perflow_wrong,
                                              nTotalConstraints,nTotalSensors)]
        info = [x[0] for x in d.itervalues()]

        # max_size = np.max(nTotalSensors)
        size = 2 # 40/max_size * nTotalSensors
        labels = [json.dumps(x,sort_keys=True, indent=4) for x in note]

        plot_scatter(nroutes,nTotalSensors,c=color,s=size,label=labels,info=info,
                     alpha=0.2,marker=marker,legend=False)

        plt.title(title)
        plt.xlabel('Number of routes')
        plt.ylabel('Number of total sensors')

    suptitle = "Size = # of sensors [solver=%s,model=%s,sparse=%s,init=%s,stat=%s,LP=^,CP=v]"
    title1 = 'Sensor selection vs config'

    if solver == 'LS':
        match_by = [('solver',solver),('sparse',sparse),('init',init)]
    elif solver == 'LSQR':
        match_by = [('solver',solver),('damp',damp)]
    if model is not None:
        match_by.append(('model',model))
    leq = [('percent flow allocated incorrectly',error_leq)]
    leq2 = [('percent flow allocated incorrectly',error_leq2)] if error_leq2 else None
    leq3 = [('percent flow allocated incorrectly',error_leq3)] if error_leq3 else None
    leq_NLPCP = [('NLPCP',max_NLPCP)] if max_NLPCP is not None else []

    sensor_param = ['use_L','use_OD','use_CP','use_LP']
    start = 1
    sensor_configs = [(True,True,True,True), (True,True,False,False),
                      (False,False,True,True),(False,False,True,False),
                      (False,False,False,True),(True,False,False,False),
                      (False,True,False,False)][start:]
    colors = ['b','g','r','c','m','y','k'][start:]
    labels = ['All','L/OD','CP/LP','CP','LP','L','OD'][start:]

    fig = plt.figure()
    legend = []
    for (config,color,label) in zip(sensor_configs,colors,labels):
        match_by_sensor = zip(sensor_param,config)
        d = filter(s,group_by=['nroutes','NLP','NCP','nOD','nLinks'],
                   match_by=match_by + match_by_sensor, leq=leq+leq_NLPCP)
        if len(d.keys()) > 0:
            print label, len(d.keys())
            plot1(d, title1, config=config, color=color, stat=stat, marker='o')
            legend.append(label)
            plot_lp(d)
    plt.legend(legend)

    for (config,color,label) in zip(sensor_configs,colors,labels):
        match_by_sensor = zip(sensor_param,config)
        d2,d3 = None,None
        if leq2 is not None:
            d2 = filter(s, group_by=['nroutes','NLP','NCP','nOD','nLinks'], geq=leq,
                        leq=leq2+leq_NLPCP, match_by=match_by+match_by_sensor)
            if len(d2.keys()) > 0:
                plt.hold(True)
                plot1(d2, title1, config=config, color=color, stat=stat,
                      marker='^')
                plot_lp(d2)
        if leq2 is not None and leq3 is not None:
            d3 = filter(s,group_by=['nroutes','NLP','NCP','nOD','nLinks'], geq=leq2,
                        leq=leq3+leq_NLPCP, match_by=match_by+match_by_sensor)
            if len(d3.keys()) > 0:
                plt.hold(True)
                plot1(d3, title1, config=config, color=color, stat=stat,
                      marker='v')
                plot_lp(d3)

    ax = plt.gca()
    ax.autoscale(True)
    fig.suptitle("%s %s" % (suptitle % (solver,model,sparse,init,stat),caption), fontsize=8)

    if disp:
        show()


def plot_sensors_vs_constraints_v2(s, init=False,sparse=True, stat='mean',
                                     caption=None,error_leq=0.01,max_NLPCP=None,
                                     model=None,solver='LS',damp=0.0,disp=True):
    """
    Plot the types and number of sensors needed to achieve 95%+ accuracy
    NEW, i.e. experiments have recorded sizes for b,d,f,g
    :param s:
    :param init:
    :param sparse:
    :param stat:
    :return:
    """
    def plot1(d, title, stat='mean'):
        nroutes = get_stats(d.itervalues(), lambda x: get_key(x,'nroutes'), stat=stat)
        blocks = get_stats(d.itervalues(), lambda x: get_key(x,'blocks'), stat=stat)
        NLP = get_stats(d.itervalues(), lambda x: get_key(x,'NLP'), stat=stat)
        NCP = get_stats(d.itervalues(), lambda x: get_key(x,'NCP'), stat=stat)
        nLPconstraints = get_stats(d.itervalues(), lambda x: get_key(x,'nLP'), stat=stat)
        nCPconstraints = get_stats(d.itervalues(), lambda x: get_key(x,'nCP'), stat=stat)

        duration = get_stats(d.itervalues(), lambda x: get_key(x,'duration'), stat=stat)
        perflow_wrong = get_stats(d.itervalues(), lambda x: get_key(x,'perflow'), stat=stat)

        note = [{'nLP constraints' : a, 'nCP constraints': b, 'NLP': c, 'NCP':e,
                 'blocks' : f, 'duration' : "{:.5f}".format(g),
                 'perflow wrong' : "{:.5f}".format(h),
                 } for (a,b,c,e,f,g,h) in zip(nLPconstraints,nCPconstraints,
                                NLP,NCP,blocks,duration,perflow_wrong)]
        info = [x[0] for x in d.itervalues()]

        colors = nroutes
        size = 2
        labels = [json.dumps(x,sort_keys=True, indent=4) for x in note]

        for (x,y,z,zz) in zip(NLP,NCP,nLPconstraints,nCPconstraints):
            plot([x,y,x+y,x],[z,zz,z+zz,z],'-k')
            plt.hold(True)
        plot_scatter(NLP+NCP,nLPconstraints+nCPconstraints,c=colors,s=size,label=labels,info=info,
                     alpha=0.2)
        plt.hold(True)
        plot_scatter(NCP,nCPconstraints,c=colors,s=size,label=labels,info=info,
                     alpha=0.2,marker='v')
        plt.hold(True)
        plot_scatter(NLP,nLPconstraints,c=colors,s=size,label=labels,info=info,
                     alpha=0.2,marker='^')

        cb = plt.colorbar()
        cb.set_alpha(1)
        cb.draw_all()
        cb.set_label('Number of routes')

        plt.title(title)
        plt.xlabel('Number of CP/LP sensors')
        plt.ylabel('Number of CP/LP constraints')
        plt.ylim(np.max([plt.ylim()[0]],0),plt.ylim()[1])
        plt.xlim(np.max([0,plt.xlim()[0]]),plt.xlim()[1])

    suptitle = "Size = fixed [solver=%s,model=%s,sparse=%s,init=%s,stat=%s,LP=^,CP=v]"
    title1 = 'NLP+NCP sensors vs constraints'

    if solver == 'LS':
        match_by = [('solver',solver),('sparse',sparse),('init',init)]
    elif solver == 'LSQR':
        match_by = [('solver',solver),('damp',damp)]
    if model is not None:
        match_by.append(('model',model))
    leq = [('percent flow allocated incorrectly',error_leq)]
    if max_NLPCP is not None:
        leq.append(('NLPCP',max_NLPCP))

    d = filter(s,group_by=['nroutes','NLP'], match_by=match_by,leq=leq)
    if len(d.keys()) > 0:
        fig = plt.figure()
        fig.suptitle("%s %s" % (suptitle % (solver,model,sparse,init,stat),caption), fontsize=8)
        plot1(d, title1, stat=stat)

    if disp:
        show()

def plot_nroutes_vs_sensors_v2(s, init=False,sparse=True, stat='mean',
                                 caption=None,error_leq=0.01,max_NLPCP=None,
                                 model=None,solver='LS',damp=0.0,disp=True):
    """
    Plot the types and number of sensors needed to achieve 95%+ accuracy
    NEW, i.e. experiments have recorded sizes for b,d,f,g
    :param s:
    :param init:
    :param sparse:
    :param stat:
    :return:
    """
    def plot2(d, title, stat='mean'):

        nroutes = get_stats(d.itervalues(), lambda x: get_key(x,'nroutes'), stat=stat)
        blocks = get_stats(d.itervalues(), lambda x: get_key(x,'blocks'), stat=stat)
        nLconstraints = get_stats(d.itervalues(), lambda x: get_key(x,'nLinks'), stat=stat)
        nODconstraints = get_stats(d.itervalues(), lambda x: get_key(x,'nOD'), stat=stat)
        perflow_wrong = get_stats(d.itervalues(), lambda x: get_key(x,'perflow'), stat=stat)
        note = [{'nL constraints': y, 'nOD constraints': q, 'blocks' : b,
                 'perflow wrong' : "{:.5f}".format(p),
                 } for (y,b,q,p) in zip(nLconstraints,blocks,nODconstraints,
                                        perflow_wrong)]
        info = [x[0] for x in d.itervalues()]

        size = 2
        labels = [json.dumps(x,sort_keys=True, indent=4) for x in note]

        plot_scatter(nroutes,nLconstraints,c='b',s=size,label=labels,info=info,
                     alpha=0.2)
        plt.hold(True)
        plot_scatter(nroutes,nODconstraints,c='r',s=size,label=labels,info=info,
                     alpha=0.2)
        plt.legend(['Link','OD'])

        plt.title(title)
        plt.xlabel('Number of routes')
        plt.ylabel('Number of sensors/constraints')
        plt.ylim(np.max([plt.ylim()[0]],0),plt.ylim()[1])
        plt.xlim(np.max([0,plt.xlim()[0]]),plt.xlim()[1])

    def plot(d, title, stat='mean'):
        # Plot information
        NLPCP = get_stats(d.itervalues(), lambda x: get_key(x,'NLPCP'), stat=stat)
        nLPCPconstraints = get_stats(d.itervalues(), lambda x: get_key(x,'nLPCP'), stat=stat)

        # Display information
        nroutes = get_stats(d.itervalues(), lambda x: get_key(x,'nroutes'), stat=stat)
        duration = get_stats(d.itervalues(), lambda x: get_key(x,'duration'), stat=stat)
        perflow_wrong = get_stats(d.itervalues(), lambda x: get_key(x,'perflow'), stat=stat)
        blocks = get_stats(d.itervalues(), lambda x: get_key(x,'blocks'), stat=stat)
        note = [{'nLPCP constraints': y, 'blocks' : b,
                  'duration' : "{:.5f}".format(q),
                  'perflow wrong' : "{:.5f}".format(p),
                  } for (y,b,q,p) in zip(nLPCPconstraints,blocks,duration,
                                         perflow_wrong)]
        info = [x[0] for x in d.itervalues()]

        # Color and size information
        colors = nLPCPconstraints
        size = 2
        labels = [json.dumps(x,sort_keys=True, indent=4) for x in note]

        plot_scatter(nroutes,NLPCP,c=colors,s=size,label=labels,info=info,
                     alpha=0.2)

        cb = plt.colorbar()
        cb.set_alpha(1)
        cb.draw_all()
        cb.set_label('Number of CP/LP constraints')
        # cb.set_label('Number of OD/Link constraints')

        plt.title(title)
        plt.xlabel('Number of routes')
        plt.ylabel('Number of CP/LP sensors')
        plt.ylim(np.max([plt.ylim()[0]],0),plt.ylim()[1])
        plt.xlim(np.max([0,plt.xlim()[0]]),plt.xlim()[1])

    suptitle = "Size = NCP [solver=%s,model=%s,sparse=%s,init=%s,stat=%s]"
    title1 = 'nroutes vs sensors'

    if solver == 'LS':
        match_by = [('solver',solver),('sparse',sparse),('init',init)]
    elif solver == 'LSQR':
        match_by = [('solver',solver),('damp',damp)]
    if model is not None:
        match_by.append(('model',model))
    leq = [('percent flow allocated incorrectly',error_leq)]
    if max_NLPCP is not None:
        leq.append(('NLPCP',max_NLPCP))

    d = filter(s,group_by=['nroutes','NLP'], match_by=match_by,leq=leq)
    if len(d.keys()) > 0:
        fig = plt.figure()
        fig.suptitle("%s %s" % (suptitle % (solver,model,sparse,init,stat),caption), fontsize=8)
        plt.subplot(121)
        plot(d, title1, stat=stat)
        plt.subplot(122)
        plot2(d, title1, stat=stat)

    if disp:
        show()

def plot_nroutes_vs_sensors(s, init=False,sparse=True, stat='mean',
                                 caption=None,error_leq=0.01,model=None,
                                 solver='LS',damp=0.0,disp=True,max_NLPCP=None):
    """
    Plot the types and number of sensors needed to achieve 95%+ accuracy
    :param s:
    :param init:
    :param sparse:
    :param stat:
    :return:
    """
    def plot(d, title, stat='mean'):
        nroutes = get_stats(d.itervalues(), lambda x: get_key(x,'nroutes'), stat=stat)
        blocks = get_stats(d.itervalues(), lambda x: get_key(x,'blocks'), stat=stat)
        NLP = get_stats(d.itervalues(), lambda x: get_key(x,'NLP'), stat=stat)
        NCP = get_stats(d.itervalues(), lambda x: get_key(x,'NCP'), stat=stat)
        perflow_wrong = get_stats(d.itervalues(), lambda x: get_key(x,'perflow'), stat=stat)
        note = [{ 'nroutes' : a, 'blocks' : b, 'NLP': c, 'NCP': e, 'perflow': f,
                  } for (a,b,c,e,f) in zip(nroutes,blocks,NLP,NCP,perflow_wrong)]
        info = [x[0] for x in d.itervalues()]

        colors = NCP
        size = blocks / nroutes
        labels = [json.dumps(x,sort_keys=True, indent=4) for x in note]

        plot_scatter(nroutes,NLP,c=colors,s=size,label=labels,info=info,
                     alpha=0.2)

        cb = plt.colorbar()
        cb.set_alpha(1)
        cb.draw_all()
        cb.set_label('Number of cellpath sensors')

        plt.title(title)
        plt.xlabel('Number of routes')
        plt.ylabel('Number of linkpath sensors')
        plt.ylim(np.max([plt.ylim()[0]],0),plt.ylim()[1])
        plt.xlim(np.max([0,plt.xlim()[0]]),plt.xlim()[1])

    suptitle = "Size = blocks / nroutes [solver=%s,model=%s,sparse=%s,init=%s,stat=%s]"
    title1 = 'nroutes vs NLP'
    if solver == 'LS':
        match_by = [('solver',solver),('sparse',sparse),('init',init)]
    elif solver == 'LSQR':
        match_by = [('solver',solver),('damp',damp)]
    if model is not None:
        match_by.append(('model',model))
    leq = [('percent flow allocated incorrectly',error_leq)]
    if max_NLPCP is not None:
        leq.append(('NLPCP',max_NLPCP))

    d = filter(s,group_by=['nroutes','NLP'], match_by=match_by,leq=leq)
    if len(d.keys()) > 0:
        fig = plt.figure()
        fig.suptitle("%s %s" % (suptitle % (solver,model,sparse,init,stat),caption), fontsize=8)
        plot(d, title1, stat=stat)

    if disp:
        show()

def plot_nroutes_vs_nblocks_init_ls(s, sparse=True, stat='mean', error_leq=1,
                                    caption=None,solver='LS',disp=True):
    def plot(d, title, stat='mean', marker=None, init=None, colorbar=True,
             error_leq=1):
        nroutes = get_stats(d.itervalues(), lambda x: get_key(x,'nroutes'), stat=stat)
        blocks = get_stats(d.itervalues(), lambda x: get_key(x,'blocks'), stat=stat)
        perflow_wrong = get_stats(d.itervalues(), lambda x: get_key(x,'perflow'), stat=stat)
        duration = get_stats(d.itervalues(), lambda x: get_key(x,'duration'), stat=stat)

        note = [{ 'nroutes' : a, 'blocks' : b, 'perflow': c,
                  } for (a,b,c) in zip(nroutes,blocks,perflow_wrong)]
        info = [x[0] for x in d.itervalues()]

        duration_capped = np.minimum(duration,600*np.ones(duration.size))
        max_size = np.max(duration_capped)
        size = 40/max_size * duration_capped
        colors = np.minimum(perflow_wrong,np.ones(perflow_wrong.size))
        labels = [json.dumps(x, sort_keys=True, indent=4) for x in note]

        plot_scatter(nroutes,blocks/nroutes,c=colors,s=size,label=labels,
                     info=info,alpha=0.2,marker=marker, vmin=0, vmax=error_leq)

        if colorbar:
            cb = plt.colorbar()
            cb.set_alpha(1)
            cb.draw_all()
            cb.set_label('nblocks')
            cb.set_label('Percent route flow error')

        plt.title(title)
        plt.xlabel('Number of routes')
        plt.ylabel('nblocks / nroutes')
        plt.ylim(np.max([plt.ylim()[0],0]),1)
        plt.xlim(np.max([0,plt.xlim()[0]]),plt.xlim()[1])

    suptitle = "[solver=%s,method=BB,sparse=%s,stat=%s; init=^, not=v]"
    title = "nroutes vs ratio to blocks"
    inits = [True,False]
    colors = ['r','b']
    markers = ['^','v']

    fig = plt.figure()
    for (init,color,marker) in zip(inits,colors,markers):
        if error_leq is None:
            d = filter(s,group_by=['nroutes','blocks'],
                       match_by=[('solver','LS'),('sparse',sparse),
                                 ('method','BB'),('init',init)])
            error_leq = 1
        else:
            d = filter(s,group_by=['nroutes','blocks'],
                       match_by=[('solver','LS'),('sparse',sparse),
                                 ('method','BB'),('init',init)],
                       leq=[('percent flow allocated incorrectly',error_leq)])
            if init:
                suptitle = "error <%d %s" % (error_leq,suptitle)
        if len(d.keys()) > 0:
            plot(d, title, stat=stat, marker=marker,
                 init=init, colorbar=init, error_leq=error_leq)
            plt.hold(True)
    fig.suptitle("%s %s" % (suptitle % (solver,sparse,stat),caption), fontsize=8)

    if disp:
        show()

def plot_nroutes_vs_nblocks(s, init=False,sparse=True, stat='mean',
                            yaxis='blocks', caption=None,model=None,
                            solver='LS',damp=0.0, disp=True, error_leq=1,
                            error_leq2=None, error_leq3=None):
    def plot_ratio(d, title, stat='mean',yaxis='blocks'):
        nroutes = get_stats(d.itervalues(), lambda x: get_key(x,'nroutes'), stat=stat)
        if yaxis=='blocks+obj':
            blocks = get_stats(d.itervalues(), lambda x: get_key(x,'blocks') + get_key(x,'nobj'), stat=stat)
        else:
            blocks = get_stats(d.itervalues(), lambda x: get_key(x,'blocks'), stat=stat)
        perflow_wrong = get_stats(d.itervalues(), lambda x: get_key(x,'perflow'), stat=stat)
        duration = get_stats(d.itervalues(), lambda x: get_key(x,'duration'), stat=stat)

        note = [{ 'nroutes' : a, yaxis : b, 'perflow': c,
                  } for (a,b,c) in zip(nroutes,blocks,perflow_wrong)]
        info = [x[0] for x in d.itervalues()]

        duration_capped = np.minimum(duration,600*np.ones(duration.size))
        max_size = np.max(duration_capped)
        size = 40/max_size * duration_capped
        colors = np.minimum(perflow_wrong,np.ones(perflow_wrong.size))
        labels = [json.dumps(x,sort_keys=True, indent=4) for x in note]

        plot_scatter(nroutes,blocks/nroutes,c=colors,s=size,label=labels,
                     info=info,alpha=0.2)

        cb = plt.colorbar()
        cb.set_alpha(1)
        cb.draw_all()
        cb.set_label('Percent route flow error')

        plt.title(title)
        plt.xlabel('Number of routes')
        plt.ylabel('%s / nroutes' % yaxis)
        plt.ylim(np.max([plt.ylim()[0],0]),1)
        plt.xlim(np.max([0,plt.xlim()[0]]),plt.xlim()[1])

    def plot(d, title, stat='mean',yaxis='blocks'):
        nroutes = get_stats(d.itervalues(), lambda x: get_key(x,'nroutes'), stat=stat)
        if yaxis=='blocks+obj':
            blocks = get_stats(d.itervalues(), lambda x: get_key(x,'blocks') + get_key(x,'nobj'), stat=stat)
        else:
            blocks = get_stats(d.itervalues(), lambda x: get_key(x,'blocks'), stat=stat)
        perflow_wrong = get_stats(d.itervalues(), lambda x: get_key(x,'perflow'), stat=stat)
        duration = get_stats(d.itervalues(), lambda x: get_key(x,'duration'), stat=stat)

        note = [{ 'nroutes' : a, yaxis : b, 'perflow': c,
                  } for (a,b,c) in zip(nroutes,blocks,perflow_wrong)]
        info = [x[0] for x in d.itervalues()]

        duration_capped = np.minimum(duration,600*np.ones(duration.size))
        max_size = np.max(duration_capped)
        size = 40/max_size * duration_capped
        colors = np.minimum(perflow_wrong,np.ones(perflow_wrong.size))
        labels = [json.dumps(x,sort_keys=True, indent=4) for x in note]

        plot_scatter(nroutes,blocks,c=colors,s=size,label=labels,info=info,
                     alpha=0.2)

        cb = plt.colorbar()
        cb.set_alpha(1)
        cb.draw_all()
        cb.set_label('Percent route flow error')

        plt.title(title)
        plt.xlabel('Number of routes')
        plt.ylabel('Number of %s' % yaxis)
        plt.ylim(np.max([plt.ylim()[0]],0),plt.ylim()[1])
        plt.xlim(np.max([0,plt.xlim()[0]]),plt.xlim()[1])
        plt.hold(True)
        xy = xrange(0,int(plt.xlim()[1]),25)
        plt.plot(xy,xy)

    suptitle = "[solver=%s,model=%s,method=%s,sparse=%s,init=%s,stat=%s]"
    title1 = "nroutes vs %s" % yaxis
    title2 = "nroutes vs ratio to %s" % yaxis
    methods = ['BB','LBFGS','DORE']

    if solver == 'LS':
        match_by = [('solver',solver),('sparse',sparse),('init',init)]
    elif solver == 'LSQR':
        match_by = [('solver',solver),('damp',damp)]
    if model is not None:
        match_by.append(('model',model))

    for method in methods:
        d = filter(s,group_by=['nroutes','blocks'],
                   match_by=match_by + [('method',method)])
        if len(d.keys()) > 0:
            fig = plt.figure()
            fig.suptitle("%s %s" % (suptitle % (solver,model,method,sparse,init,stat),caption), fontsize=8)
            plt.subplot(121)
            plot(d, title1, stat=stat, yaxis=yaxis)
            plt.subplot(122)
            plot_ratio(d, title2, stat=stat, yaxis=yaxis)

    if disp:
        show()

if __name__ == "__main__":

    # Plot configuration
    init = False
    sparse = False
    yaxis = 'blocks'
    error = 0.10
    error2 = 0.30
    error3 = 0.50

    scenarios, scenarios_all_links, scenarios_v2, scenarios_v3 = load_results()

    plot_solver_comparison(scenarios_v3, sparse=False, caption='',
                       error_leq=error, error_leq2=error2, model='UE',
                       error_leq3=error3, max_NLPCP=100)

    # PLOT LS vs LSQ



