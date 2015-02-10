
import pickle
import time

import numpy as np
from random import randint

import config as c
from synthetic_traffic.sensors.SensorConfiguration import SensorConfiguration

def save(fname, TN):
    with open(fname, 'w') as f:
        pickle.dump(TN, f)

def generate_sensor_configurations(num_links, num_ODs, num_cellpath_NBs,
                                   num_cellpath_NLs, num_cellpath_NSs,
                                   num_linkpaths, times=1, myseed=None):
    for i in range(times):
        for num_link in num_links:
            for num_OD in num_ODs:
                for num_cellpath_NB in num_cellpath_NBs:
                    for num_cellpath_NL in num_cellpath_NLs:
                        for num_cellpath_NS in num_cellpath_NSs:
                            for num_linkpath in num_linkpaths:
                                SC = SensorConfiguration(num_link=num_link,
                                               num_OD=num_OD,
                                               num_cellpath_NB=num_cellpath_NB,
                                               num_cellpath_NL=num_cellpath_NL,
                                               num_cellpath_NS=num_cellpath_NS,
                                               num_linkpath=num_linkpath)
                                t = int(time.time())
                                r = randint(1e5,999999)
                                fname = "%s/SC_%s_%s.pkl" % (c.SC_DIR,t,r)
                                save(fname, SC)

if __name__ == "__main__":
    num_links = [0, np.inf]
    num_ODs = [0, np.inf]
    num_cellpath_NBs = range(0,300,30)
    num_cellpath_NLs = [0, 100, np.inf]
    num_cellpath_NSs = [0]
    num_linkpaths = range(0,300,30)
    myseed = 2347234328
    times = 1

    generate_sensor_configurations(num_links,num_ODs,num_cellpath_NBs,
                                   num_cellpath_NLs,num_cellpath_NSs,
                                   num_linkpaths,times=times,myseed=myseed)


