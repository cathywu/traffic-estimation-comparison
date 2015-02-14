
import numpy as np

import config as c
from synthetic_traffic.sensors.SensorConfiguration import SensorConfiguration

from scenario_utils import save

def generate_sensor_configurations(num_links, num_ODs, num_cellpath_NBs,
                                   num_cellpath_NLs, num_cellpath_NSs,
                                   num_linkpaths, times=1, myseed=None,
                                   prefix='%s/SC'):
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
                                save(SC, prefix=prefix % c.SC_DIR)

if __name__ == "__main__":
    num_links = [0, np.inf]
    num_ODs = [0, np.inf]
    num_cellpath_NBs = [5,7,8,9,14,16,29,38,51] # range(0,300,30)
    num_cellpath_NLs = [0] # [0, 100, np.inf]
    num_cellpath_NSs = [0]
    num_linkpaths = [1,2,3,5,6,8] # range(0,300,30)
    myseed = 2347234328
    times = 1

    generate_sensor_configurations(num_links=num_links,num_ODs=num_ODs,
                                   num_cellpath_NBs=num_cellpath_NBs,
                                   num_cellpath_NLs=num_cellpath_NLs,
                                   num_cellpath_NSs=num_cellpath_NSs,
                                   num_linkpaths=num_linkpaths,
                                   times=times,myseed=myseed)


