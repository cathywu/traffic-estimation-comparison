import config as c
from synthetic_traffic.networks.GridNetwork import GridNetwork
from synthetic_traffic.networks.EquilibriumNetwork import EquilibriumNetwork
from synthetic_traffic.sensors.SensorConfiguration import SensorConfiguration

from scenario_utils import save


def generate_grid_networks(nrows, ncols, nodroutes, times=1, myseed=None,
                           prefix='%s/TN_Grid'):
    for i in range(times):
        for nrow in nrows:
            for ncol in ncols:
                for nodroute in nodroutes:
                    TN = GridNetwork(ncol=ncol,nrow=nrow,nodroutes=nodroute,
                                     myseed=myseed)
                    save(TN, prefix=prefix % c.TN_DIR)
                    TN = GridNetwork(ncol=ncol,nrow=nrow,nodroutes=nodroute,
                                     myseed=myseed, concentration=0.1)
                    save(TN, prefix=prefix % c.TN_DIR)

def generate_equilibrium_networks(SOs=(False),path=None, prefix='%s/TN_EQ'):
    for SO in SOs:
        TN = EquilibriumNetwork(SO=SO,path=path)
        save(TN, prefix=prefix % c.TN_DIR)

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

