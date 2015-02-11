
import time

import config as c
from synthetic_traffic.networks.GridNetwork import GridNetwork
from synthetic_traffic.networks.EquilibriumNetwork import EquilibriumNetwork

from scenario_utils import save

def generate_grid_networks(nrows, ncols, nodroutes, times=1, myseed=None):
    for i in range(times):
        for nrow in nrows:
            for ncol in ncols:
                for nodroute in nodroutes:
                    TN = GridNetwork(ncol=ncol,nrow=nrow,nodroutes=nodroute,
                                     myseed=myseed)
                    save(TN, prefix="%s/TN_Grid" % c.TN_DIR)

def generate_equilibrium_networks(SOs=(False),path=None):
    for SO in SOs:
        TN = EquilibriumNetwork(SO=SO,path=path)
        fname = "%s/networks/TN_EQ_%s.pkl" % (c.DATA_DIR,time.time())
        save(fname, TN)

if __name__ == "__main__":
    myseed = 2347234328
    times = 1

    nrows = range(1,11)
    ncols = range(1,11)
    nodroutes = [15]

    generate_grid_networks(nrows,ncols,nodroutes,times=times,myseed=myseed)

    SOs = [True, False]
    EQ_network_path = 'los_angeles_data_2.mat'
    # generate_equilibrium_networks(SOs=SOs,path=EQ_network_path)


