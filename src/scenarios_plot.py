from __future__ import division
import ipdb
import json
import os
from matplotlib.pyplot import figure, show
import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from pylab import connect, scatter, plot
from AnnoteFinder import AnnoteFinder
from Scenario import to_np

def filter_valid(d):
    if 'error' in d:
        return False
    if 'duration' not in d:
        return False
    return True

def filter_v2(d):
    if 'nLinks' not in d and 'nOD' not in d and 'nCP' not in d and 'nLP' not in d:
        return False
    return True

def filter_v3(d):
    if 'use_L' not in d['params']:
        return False
    return True

def isp(d,name=None,match=None):
    if name in d['params'] and d['params'][name] == match:
        return True
    return False

def filter_all_links(d):
    p = d['params']
    if 'all_links' in p and p['all_links'] == True and p['NLP'] == 0 and \
                    p['NL'] == 0 and p['NB'] == 0 and p['NS'] == 0:
        return True
    return False

def plot_scatter(x,y,c=None,s=None,label=None,info=None,alpha=1.0,marker='o',
                 vmin=None,vmax=None):
    # x, y, c, s = rand(4, 100)
    scatter(x, y, 100*s, c, alpha=alpha,marker=marker,vmin=vmin,vmax=vmax)
    #fig.savefig('pscoll.eps')
    if label is not None:
        af =  AnnoteFinder(x,y,label,info=info)
        connect('button_press_event', af)

def filter(s,group_by=[],match_by=[],geq=[],leq=[]):
    d = {}
    valid = ['nroutes','nsensors','blocks','percent flow allocated incorrectly',
             'NLPCP','use_L','use_OD','use_CP','use_LP']
    for x in s:
        match = True
        for (param,value) in match_by:
            if param in valid and get_key(x,param) != value or \
                    param in x['params'] and get_key(x,param) != value:
                match = False
                break
        if match == False:
            continue
        for (param,value) in geq:
            try:
                if param in valid and get_key(x,param) < value or \
                        param in x['params'] and get_key(x,param) < value:
                    match = False
                    break
            except:
                ipdb.set_trace()
        if match == False:
            continue
        for (param,value) in leq:
            if param in valid and get_key(x,param) > value or \
                    param in x['params'] and get_key(x,param) > value:
                match = False
                break
        if match == False:
            continue
        try:
            key = frozenset([(group,get_key(x,group)) for group in group_by])
        except TypeError:
            ipdb.set_trace()
        if key in d:
            d[key].append(x)
        else:
            d[key] = [x]
    return d

def get_key(d,key):
    if key == 'nroutes':
        return d['AA'][1]
    elif key == 'nobj':
        return d['AA'][0]
    elif key in ['perflow','percent flow allocated incorrectly']:
        return _get_per_flow(d)
    elif key == 'blocks':
        return _get_blocks(d)
    elif key == 'NCP':
        return d['params']['NB'] + d['params']['NS'] + d['params']['NL']
    elif key == 'NLPCP':
        return d['params']['NLP'] + get_key(d,'NCP')
    elif key == 'nLPCP':
        return d['nLP'] + d['nCP']
    elif key in ['nLinks','nOD','nCP','nLP','duration']:
        # CAUTION: only works for v2+
        return d[key] if key in d else 0
    elif key in ['use_L','use_OD','use_CP','use_LP']:
        # CAUTION: only works for v2+
        return d['params'][key] if key in d['params'] else True
    else:
        return d['params'][key]

def _get_blocks(d):
    if d['blocks'] is None:
        return 0
    return d['blocks'][0]

def _get_per_flow(d):
    if type(d['percent flow allocated incorrectly']) == type([]):
        return d['percent flow allocated incorrectly'][-1]
    else:
        return d['percent flow allocated incorrectly']

def get_stats(xs,f,stat='mean'):
    return np.array([_get_stat(x,f,stat=stat) for x in xs])

def _get_stat(l, f, stat='first'):
    if stat=='first':
        return f(l[0])
    elif stat=='mean':
        return np.mean([f(x) for x in l])
    elif stat=='median':
        return np.median([f(x) for x in l])
    else:
        print "Error: stat %s not found" % stat

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

def plot_sensors_vs_configs_v3(s, init=False, sparse=True, stat='mean',
                               caption=None, error_leq=1, error_leq2=None,
                               error_leq3=None, max_NLPCP=None, model=None,
                               solver='LS', damp=0.0, disp=True):
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
        nTotalConstraints = config[0] * nLconstraints + config[1] * nODconstraints \
                            + config[2] * nCPconstraints + config[3] * nLPconstraints
        nTotalSensors = config[0] * nLconstraints + config[1] * nODconstraints \
                        + config[2] * NCP + config[3] * NLP

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
                     alpha=0.2,marker=marker)

        plt.title(title)
        plt.xlabel('Number of routes')
        plt.ylabel('Number of total sensors')
        plt.ylim(np.max([plt.ylim()[0]],0),plt.ylim()[1])
        plt.xlim(np.max([0,plt.xlim()[0]]),plt.xlim()[1])

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

    sensor_configs = [(True,True,True,True), (True,True,False,False),
                      (False,False,True,True),(False,False,True,False),
                      (False,False,False,True),(True,False,False,False),
                      (False,True,False,False)]
    sensor_param = ['use_L','use_OD','use_CP','use_LP']
    colors = ['b','g','r','c','m','y','k']
    labels = ['All','L/OD','CP/LP','CP','LP','L','OD']

    fig = plt.figure()
    legend = []
    for (config,color,label) in zip(sensor_configs,colors,labels):
        match_by_sensor = zip(sensor_param,config)
        d = filter(s,group_by=['nroutes','NLP'],
                   match_by=match_by + match_by_sensor, leq=leq+leq_NLPCP)
        if len(d.keys()) > 0:
            print label, len(d.keys())
            plot1(d, title1, config=config, color=color, stat=stat, marker='o')
            legend.append(label)
    plt.legend(legend)

    for (config,color,label) in zip(sensor_configs,colors,labels):
        match_by_sensor = zip(sensor_param,config)
        if leq2 is not None:
            d2 = filter(s, group_by=['nroutes','NLP'],leq=leq2+leq_NLPCP,
                        match_by=match_by + match_by_sensor, geq=leq)
            if len(d2.keys()) > 0:
                plt.hold(True)
                plot1(d2, title1, config=config, color=color, stat=stat, marker='^')
        if leq2 is not None and leq3 is not None:
            d3 = filter(s,group_by=['nroutes','NLP'], leq=leq3+leq_NLPCP, geq=leq2,
                        match_by=match_by+match_by_sensor)
            if len(d3.keys()) > 0:
                plt.hold(True)
                plot1(d3, title1, config=config, color=color, stat=stat, marker='v')

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
        cb.set_label('nblocks / nroutes')

        plt.title(title)
        plt.xlabel('Number of routes')
        plt.ylabel('Number of linkpath sensors')
        plt.ylim(np.max([plt.ylim()[0]],0),plt.ylim()[1])
        plt.xlim(np.max([0,plt.xlim()[0]]),plt.xlim()[1])

    suptitle = "Size = NCP [solver=%s,model=%s,sparse=%s,init=%s,stat=%s]"
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
    import config as c
    files = os.listdir(c.RESULT_DIR)
    curr_input = None

    scenarios = []
    scenarios_all_links = []
    scenarios_v2 = []
    scenarios_v3 = []

    for file in files:
        filename = "%s/%s" % (c.RESULT_DIR,file)
        if os.path.isdir(filename):
            continue
        print file
        with open(filename) as out:
            for line in out:
                if line[0] != '{':
                    continue

                try:
                    d = json.loads(line)
                except ValueError:
                    # ipdb.set_trace()
                    continue
                if not filter_valid(d):
                    continue
                # Minor correction: get rid of 'var'
                if 'var' in d:
                    d['var'] = to_np(d['var'])
                # Minor correction: for <v2, default NL/NB/NS=100
                if not filter_v3(d): # FIXME add new versions as needed
                    if d['params']['NL'] == 0:
                        d['params']['NL'] = 100
                    if d['params']['NB'] == 0:
                        d['params']['NB'] = 100
                    if d['params']['NS'] == 0:
                        d['params']['NS'] = 100

                scenarios.append(d)
                if filter_all_links(d):
                    scenarios_all_links.append(d)
                if filter_v2(d):
                    scenarios_v2.append(d)
                if filter_v3(d):
                    scenarios_v3.append(d)

    # Plot configuration
    init = False
    sparse = False
    yaxis = 'blocks'
    error = 0.10
    error2 = 0.30
    error3 = 0.50

    # caption = """This plot compares accuracy (%% error), duration, and the number of blocks, as the number of routes considered grows (which is
    # a function of the network and nroutes parameter. To the left, we have on the yaxis the number of %s; the right plot shows the same information but displays a
    # ratio of the %s over the number of routes instead.""" % (yaxis, yaxis)
    # plot_nroutes_vs_nblocks(scenarios, init=init, sparse=sparse,
    #                            yaxis=yaxis, caption=caption, solver='LS', model='P')
    # caption = """This plot is the same as the previous, but displays results for ONLY the LS/UE experiments"""
    # plot_nroutes_vs_nblocks(scenarios, init=init, sparse=sparse,
    #                            yaxis=yaxis, caption=caption, solver='LS', model='UE')
    # caption = """This plot is the same as the previous, but displays results for ONLY the LS/SO experiments"""
    # plot_nroutes_vs_nblocks(scenarios, init=init, sparse=sparse,
    #                            yaxis=yaxis, caption=caption, solver='LS', model='SO')
    # # FIXME all_links does nothing for P networks
    # caption = """This plot is also the same, but displays results for only experiments flagged "all_links", which unforunately only makes a difference for
    # the UE/SO experiments."""
    # plot_nroutes_vs_nblocks(scenarios_all_links, init=init, sparse=sparse,
    #                            yaxis=yaxis, caption=caption)
    # caption = """This plot first filters LS/P experiments that get %0.2f+ route flow accuracy, then looks at the number of selected sensors (LP on the yaxis,
    # CP encoded in color). Unfortunately, this doesn't give a sense of how much information is from OD pairs, links, and also how much information is
    # provided by the LP/CP sensors in aggregate""" % error
    # plot_nroutes_vs_sensors(scenarios, init=init, sparse=sparse,
    #                              caption=caption,error_leq=error,solver='LS',model='P')
    # caption = """This plot is the same as the previous but better (but with less data for now) because we have also recorded the number of constraints that
    # come from the various sensors. CAUTION: it is the number of resulting constraints, NOT the number of sensors of each type, that is displayed."""
    # plot_nroutes_vs_sensors_v2(scenarios_v2, init=False, sparse=sparse,
    #                              caption=caption,error_leq=error,solver='LS',model='P',disp=False)
    # plot_nroutes_vs_sensors_v2(scenarios_v3, init=False, sparse=sparse,
    #                            caption=caption,error_leq=error,solver='LSQR',model='P')
    # # FIXME I'm running more experiments in the empty space, check back in a bit
    # caption = """For LS experiments that perform well, this plot shows the makeup of LP/CP sensors"""
    # plot_sensors_vs_constraints_v2(scenarios_v2, init=False, sparse=sparse,
    #                                  caption=caption,error_leq=error,max_NLPCP=350,
    #                                  solver='LS',disp=False)
    # plot_sensors_vs_constraints_v2(scenarios_v2, init=False, sparse=sparse,
    #                                    caption=caption,error_leq=error,max_NLPCP=350,
    #                                    solver='LSQR')
    # ############
    # caption = """Now we start to look at if initializing the LS solver with a guess based off of the equality constraint yields any performance improvement.
    # The answer looks like yes with regards to accuracy but no with regards to runtime. I think this is bad news."""
    # plot_nroutes_vs_nblocks_init_ls(scenarios_ls, sparse=sparse,
    #                                 caption=caption)
    # ############
    # caption = """This is basically the same plot, but we filter for experiments that are %0.2f+ accurate in route flow""" % error
    # plot_nroutes_vs_nblocks_init_ls(scenarios_ls, sparse=sparse, error_leq=error,
    #                                 caption=caption)
    caption = """[ALL] This is the first plot with LSQR results; it shows the total number of constraints for experiments with a %0.2f+ route flow accuracy,
    colored by the sensor configuration, and sized by the number of actual sensors"""
    plot_sensors_vs_configs_v3(scenarios_v3, sparse=sparse,solver='LSQR',
                               caption=caption,max_NLPCP=100,disp=False)
    caption = """[ACCURATE] This is the first plot with LSQR results; it shows the total number of constraints for experiments with a %0.2f+ route flow accuracy,
    colored by the sensor configuration, and sized by the number of actual sensors"""
    plot_sensors_vs_configs_v3(scenarios_v3, sparse=sparse,solver='LSQR',
                                caption=caption,error_leq=error,error_leq2=error2,
                                error_leq3=error3,max_NLPCP=100)
    caption = """Same but for LS"""
    plot_sensors_vs_configs_v3(scenarios_v2, sparse=sparse,solver='LS',
                               caption=caption,error_leq=error,error_leq2=error2,
                               error_leq3=error3,max_NLPCP=350)

    # PLOT LS vs LSQ



