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

def filter_valid(d):
    if 'error' in d:
        return False
    if 'duration' not in d:
        return False
    return True

def filter_new(d):
    if 'nLinks' not in d:
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
    valid = ['nroutes','nsensors','blocks','percent flow allocated incorrectly','NLPCP']
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
        return d['AA'][0]
    elif key == 'nsensors':
        return d['AA'][1]
    elif key == 'percent flow allocated incorrectly':
        return get_per_flow(d)
    elif key == 'blocks':
        return get_blocks(d)
    elif key == 'NLPCP':
        return d['params']['NLP'] + d['params']['NB'] + d['params']['NS'] + d['params']['NL']
    else:
        return d['params'][key]

def get_blocks(d):
    if d['blocks'] is None:
        return 0
    return d['blocks'][0]

def get_per_flow(d):
    if type(d['percent flow allocated incorrectly']) == type([]):
        return d['percent flow allocated incorrectly'][-1]
    else:
        return d['percent flow allocated incorrectly']

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

def get_stat(l, f, stat='first'):
    if stat=='first':
        return f(l[0])
    elif stat=='mean':
        return np.mean([f(x) for x in l])
    elif stat=='median':
        return np.median([f(x) for x in l])
    else:
        print "Error: stat %s not found" % stat

def plot_sensors_vs_constraints_ls_new(s, init=False,sparse=True, stat='mean',
                                     caption=None):
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
        nroutes = np.array([get_stat(x,lambda x: x['AA'][1],stat=stat) \
                            for x in d.itervalues()])
        blocks = np.array([get_stat(x,lambda x: get_blocks(x),stat=stat) \
                           for x in d.itervalues()])
        NLP = np.array([get_stat(x,lambda x: x['params']['NLP'],stat=stat) \
                        for x in d.itervalues()])
        NCP = np.array([get_stat(x,lambda x: x['params']['NB']+x['params']['NL'],
                                 stat=stat) for x in d.itervalues()])
        nLPconstraints = np.array([get_stat(x,lambda x: x['nLP'],stat=stat) \
                                     for x in d.itervalues()])
        nCPconstraints = np.array([get_stat(x,lambda x: x['nCP'],stat=stat) \
                                   for x in d.itervalues()])
        nLconstraints = np.array([get_stat(x,lambda x: x['nLinks'],stat=stat) \
                                    for x in d.itervalues()])
        nODconstraints = np.array([get_stat(x,lambda x: x['nOD'],stat=stat) \
                                   for x in d.itervalues()])
        note = [{ 'x' : x[0]['AA'][1], 'y' : get_blocks(x[0]),
                  'duration' : "{:.5f}".format(x[0]['duration']),
                  'perflow wrong' : "{:.5f}".format(get_per_flow(x[0])),
                  } for x in d.itervalues()]
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

    suptitle = "Size = fixed [sparse=%s,init=%s,stat=%s,LP=^,CP=v]"
    title1 = 'NLP+NCP sensors vs constraints'
    d = filter(s,group_by=['nroutes','NLP'],
               match_by=[('solver','LS'),('sparse',sparse),('init',init)],
               leq=[('percent flow allocated incorrectly',0.01),('NLPCP',350)])
    if len(d.keys()) > 0:
        fig = plt.figure()
        fig.suptitle("%s %s" % (suptitle % (sparse,init,stat),caption), fontsize=8)
        plot1(d, title1, stat=stat)

    show()

def plot_nroutes_vs_sensors_ls_P_new(s, init=False,sparse=True, stat='mean',
                                 caption=None):
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
        nroutes = np.array([get_stat(x,lambda x: x['AA'][1],stat=stat) \
                            for x in d.itervalues()])
        blocks = np.array([get_stat(x,lambda x: get_blocks(x),stat=stat) \
                           for x in d.itervalues()])
        NLP = np.array([get_stat(x,lambda x: x['params']['NLP'],stat=stat) \
                        for x in d.itervalues()])
        NCP = np.array([get_stat(x,lambda x: x['params']['NB']+x['params']['NL'],
                                 stat=stat) for x in d.itervalues()])
        nLPCPconstraints = np.array([get_stat(x,lambda x: x['nLP'] + x['nCP'],stat=stat) \
                                     for x in d.itervalues()])
        nLconstraints = np.array([get_stat(x,lambda x: x['nLinks'],stat=stat) \
                                  for x in d.itervalues()])
        nODconstraints = np.array([get_stat(x,lambda x: x['nOD'],stat=stat) \
                                   for x in d.itervalues()])
        note = [{ 'x' : x[0]['AA'][1], 'y' : get_blocks(x[0]),
                  'duration' : "{:.5f}".format(x[0]['duration']),
                  'perflow wrong' : "{:.5f}".format(get_per_flow(x[0])),
                  } for x in d.itervalues()]
        info = [x[0] for x in d.itervalues()]

        size = blocks / nroutes
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
        NLP = np.array([get_stat(x,lambda x: x['params']['NLP'],stat=stat) \
                        for x in d.itervalues()])
        NCP = np.array([get_stat(x,lambda x: x['params']['NB']+x['params']['NL'],
                                 stat=stat) for x in d.itervalues()])
        nLPCPconstraints = np.array([get_stat(x,lambda x: x['nLP'] + x['nCP'],stat=stat) \
                                     for x in d.itervalues()])

        # Display information
        nroutes = np.array([get_stat(x,lambda x: x['AA'][1],stat=stat) \
                            for x in d.itervalues()])
        duration = np.array([get_stat(x,lambda x: x['duration'],stat=stat) \
                             for x in d.itervalues()])
        perflow_wrong = np.array([get_stat(x,lambda x: get_per_flow(x),stat=stat) \
                                  for x in d.itervalues()])
        blocks = np.array([get_stat(x,lambda x: get_blocks(x),stat=stat) \
                           for x in d.itervalues()])
        note = [{'nLPCP constraints': y, 'blocks' : b,
                  'duration' : "{:.5f}".format(d),
                  'perflow wrong' : "{:.5f}".format(p),
                  } for (x,y,b,d,p) in zip(d.itervalues(),nLPCPconstraints,
                                           blocks,duration,perflow_wrong)]
        info = [x[0] for x in d.itervalues()]

        # Color and size information
        colors = nLPCPconstraints
        size = blocks / nroutes
        labels = [json.dumps(x,sort_keys=True, indent=4) for x in note]

        plot_scatter(nroutes,NLP+NCP,c=colors,s=size,label=labels,info=info,
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

    suptitle = "Size = NCP [sparse=%s,init=%s,stat=%s]"
    title1 = 'nroutes vs sensors'
    d = filter(s,group_by=['nroutes','NLP'],
               match_by=[('solver','LS'),('sparse',sparse),('init',init)],
               leq=[('percent flow allocated incorrectly',0.01)])
    if len(d.keys()) > 0:
        fig = plt.figure()
        fig.suptitle("%s %s" % (suptitle % (sparse,init,stat),caption), fontsize=8)
        plt.subplot(121)
        plot(d, title1, stat=stat)
        plt.subplot(122)
        plot2(d, title1, stat=stat)

    show()

def plot_nroutes_vs_sensors_ls_P(s, init=False,sparse=True, stat='mean',
                                 caption=None):
    """
    Plot the types and number of sensors needed to achieve 95%+ accuracy
    :param s:
    :param init:
    :param sparse:
    :param stat:
    :return:
    """
    def plot(d, title, stat='mean'):
        nroutes = np.array([get_stat(x,lambda x: x['AA'][1],stat=stat) \
                            for x in d.itervalues()])
        blocks = np.array([get_stat(x,lambda x: get_blocks(x),stat=stat) \
                           for x in d.itervalues()])
        NLP = np.array([get_stat(x,lambda x: x['params']['NLP'],stat=stat) \
                        for x in d.itervalues()])
        NCP = np.array([get_stat(x,lambda x: x['params']['NB']+x['params']['NL'],
                                 stat=stat) for x in d.itervalues()])
        note = [{ 'x' : x[0]['AA'][1], 'y' : get_blocks(x[0]),
                  'duration' : "{:.5f}".format(x[0]['duration']),
                  'perflow wrong' : "{:.5f}".format(get_per_flow(x[0])),
                  } for x in d.itervalues()]
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

    suptitle = "Size = NCP [sparse=%s,init=%s,stat=%s]"
    title1 = 'nroutes vs NLP'
    d = filter(s,group_by=['nroutes','NLP'],
               match_by=[('solver','LS'),('sparse',sparse),('init',init)],
               leq=[('percent flow allocated incorrectly',0.01)])
    if len(d.keys()) > 0:
        fig = plt.figure()
        fig.suptitle("%s %s" % (suptitle % (sparse,init,stat),caption), fontsize=8)
        plot(d, title1, stat=stat)

    show()

def plot_nroutes_vs_nblocks_init_ls(s, sparse=True, stat='mean', error_leq=1,
                                    caption=None):
    def plot(d, title, stat='mean', marker=None, init=None, colorbar=True,
             error_leq=1):
        nroutes = np.array([get_stat(x,lambda y: y['AA'][1],stat=stat) for x in d.itervalues()])
        blocks = np.array([get_stat(x,lambda y: get_blocks(y),stat=stat) for x in d.itervalues()])
        note = [{ 'x' : x[0]['AA'][1], 'y' : get_blocks(x[0]),
                  'duration' : "{:.5f}".format(x[0]['duration']),
                  'perflow wrong' : "{:.5f}".format(get_per_flow(x[0])),
                  } for x in d.itervalues()]
        info = [x[0] for x in d.itervalues()]

        size = np.array([40*x[0]['duration']+1 for x in d.itervalues()])
        max_size = np.max(size)
        size = 40/max_size * size
        colors = [np.min([1,get_per_flow(x[0])]) for x in d.itervalues()]
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

    suptitle = "[method=BB,sparse=%s,stat=%s; init=^, not=v]"
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
    fig.suptitle("%s %s" % (suptitle % (sparse,stat),caption), fontsize=8)
    show()

def plot_nroutes_vs_nblocks_ls(s, init=False,sparse=True, stat='mean',
                               yaxis='blocks', caption=None):
    def plot_ratio(d, title, stat='mean',yaxis='blocks'):
        nroutes = np.array([get_stat(x,lambda y: y['AA'][1],stat=stat) for x \
                            in d.itervalues()])
        if yaxis=='blocks+obj':
            blocks = np.array([get_stat(x,lambda y: get_blocks(y) + y['AA'][0],
                                        stat=stat) for x in d.itervalues()])
        else:
            blocks = np.array([get_stat(x,lambda y: get_blocks(y),stat=stat) \
                               for x in d.itervalues()])
        note = [{ 'x' : y, 'y' : z,
                  'duration' : "{:.5f}".format(x[0]['duration']),
                  'perflow wrong' : "{:.5f}".format(get_per_flow(x[0])),
                  } for (y,z,x) in zip(nroutes,blocks,d.itervalues())]
        info = [x[0] for x in d.itervalues()]

        colors = [np.min([1,get_per_flow(x[0])]) for x in d.itervalues()]
        size = [40*x[0]['duration']+1 for x in d.itervalues()]
        labels = [json.dumps(x,sort_keys=True, indent=4) for x in note]

        plot_scatter(nroutes,blocks/nroutes,c=colors,s=size,label=labels,
                     info=info,alpha=0.008)

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
        nroutes = np.array([get_stat(x,lambda x: x['AA'][1],stat=stat) for x \
                            in d.itervalues()])
        if yaxis=='blocks+obj':
            blocks = np.array([get_stat(x,lambda y: get_blocks(y) + y['AA'][0],
                                        stat=stat) for x in d.itervalues()])
        else:
            blocks = np.array([get_stat(x,lambda y: get_blocks(y),stat=stat) \
                               for x in d.itervalues()])
        note = [{ 'x' : y, 'y' : z,
                  'duration' : "{:.5f}".format(x[0]['duration']),
                  'perflow wrong' : "{:.5f}".format(get_per_flow(x[0])),
                  } for (y,z,x) in zip(nroutes,blocks,d.itervalues())]
        info = [x[0] for x in d.itervalues()]

        colors = [np.min([1,get_per_flow(x[0])]) for x in d.itervalues()]
        size = [40*x[0]['duration']+1 for x in d.itervalues()]
        labels = [json.dumps(x,sort_keys=True, indent=4) for x in note]

        plot_scatter(nroutes,blocks,c=colors,s=size,label=labels,info=info,
                    alpha=0.008)

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

    suptitle = "[%s,sparse=%s,init=%s,stat=%s]"
    title1 = "nroutes vs %s" % yaxis
    title2 = "nroutes vs ratio to %s" % yaxis
    methods = ['BB','LBFGS','DORE']

    for method in methods:
        d = filter(s,group_by=['nroutes','blocks'],
                   match_by=[('sparse',sparse),('method',method),('init',init)])
        if len(d.keys()) > 0:
            fig = plt.figure()
            fig.suptitle("%s %s" % (suptitle % (method,sparse,init,stat),caption), fontsize=8)
            plt.subplot(121)
            plot(d, title1, stat=stat, yaxis=yaxis)
            plt.subplot(122)
            plot_ratio(d, title2, stat=stat, yaxis=yaxis)

    show()

if __name__ == "__main__":
    import config as c
    files = os.listdir(c.RESULT_DIR)
    curr_input = None

    scenarios_ls_P = []
    scenarios_ls_UE = []
    scenarios_ls_SO = []
    scenarios_all_links = []
    scenarios_new = []

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
                if isp(d,'solver','LS') and isp(d,'model','P'):
                    scenarios_ls_P.append(d)
                if isp(d,'solver','LS') and isp(d,'model','UE'):
                    scenarios_ls_UE.append(d)
                if isp(d,'solver','LS') and isp(d,'model','SO'):
                    scenarios_ls_SO.append(d)
                if filter_all_links(d):
                    scenarios_all_links.append(d)
                if filter_new(d):
                    scenarios_new.append(d)

    # plot_nroutes_vs_nblocks_ls_P(scenarios_ls_P, init=True, sparse=True)

    # Collect groups of scenarios
    scenarios_ls = []
    scenarios_ls.extend(scenarios_ls_P)
    scenarios_ls.extend(scenarios_ls_UE)
    scenarios_ls.extend(scenarios_ls_SO)

    # Plot configuration
    sparse = False
    yaxis = 'blocks'

    # caption = """This plot compares accuracy (%% error), duration, and the number of blocks, as the number of routes considered grows (which is
    # a function of the network and nroutes parameter. To the left, we have on the yaxis the number of %s; the right plot shows the same information but displays a
    # ratio of the %s over the number of routes instead.""" % (yaxis, yaxis)
    # plot_nroutes_vs_nblocks_ls(scenarios_ls, init=False, sparse=sparse,
    #                            yaxis=yaxis, caption=caption)
    # caption = """This plot is the same as the previous, but displays results for ONLY the LS/UE experiments"""
    # plot_nroutes_vs_nblocks_ls(scenarios_ls_UE, init=False, sparse=sparse,
    #                            yaxis=yaxis, caption=caption)
    # caption = """This plot is the same as the previous, but displays results for ONLY the LS/SO experiments"""
    # plot_nroutes_vs_nblocks_ls(scenarios_ls_SO, init=False, sparse=sparse,
    #                            yaxis=yaxis, caption=caption)
    # # FIXME all_links does nothing for P networks
    # caption = """This plot is also the same, but displays results for only experiments flagged "all_links", which unforunately only makes a difference for
    # the UE/SO experiments."""
    # plot_nroutes_vs_nblocks_ls(scenarios_all_links, init=False, sparse=sparse,
    #                            yaxis=yaxis, caption=caption)
    # caption = """This plot first filters LS/P experiments that get 99%%+ route flow accuracy, then looks at the number of selected sensors (LP on the yaxis,
    # CP encoded in color). Unfortunately, this doesn't give a sense of how much information is from OD pairs, links, and also how much information is
    # provided by the LP/CP sensors in aggregate"""
    # plot_nroutes_vs_sensors_ls_P(scenarios_ls_P, init=False, sparse=sparse,
    #                              caption=caption)
    caption = """This plot is the same as the previous but better (but with less data for now) because we have also recorded the number of constraints that
    come from the various sensors. CAUTION: it is the number of resulting constraints, NOT the number of sensors of each type, that is displayed."""
    plot_nroutes_vs_sensors_ls_P_new(scenarios_new, init=False, sparse=sparse,
                                 caption=caption)
    # # FIXME I'm running more experiments in the empty space, check back in a bit
    # caption = """For LS experiments that perform well, this plot shows the makeup of LP/CP sensors"""
    # plot_sensors_vs_constraints_ls_new(scenarios_new, init=False, sparse=sparse,
    #                                  caption=caption)
    # ############
    # caption = """Now we start to look at if initializing the LS solver with a guess based off of the equality constraint yields any performance improvement.
    # The answer looks like yes with regards to accuracy but no with regards to runtime. I think this is bad news."""
    # plot_nroutes_vs_nblocks_init_ls(scenarios_ls, sparse=sparse,
    #                                 caption=caption)
    # ############
    # caption = """This is basically the same plot, but we filter for experiments that are 99%%+ accurate in route flow"""
    # plot_nroutes_vs_nblocks_init_ls(scenarios_ls, sparse=sparse, error_leq=0.01,
    #                                 caption=caption)


