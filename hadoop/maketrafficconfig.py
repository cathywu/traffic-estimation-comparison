import sys

if len(sys.argv) != 2:
	print "the number of arguments is", len(sys.argv) - 1, " when expected 1." 
	raise Exception

base = sys.argv[1].strip()

config = '''ACCEPTED_LOG_LEVELS = ['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'WARN']

DATA_DIR = '{0}/traffic-estimation/comparison/data'
TN_DIR = '%s/networks' % DATA_DIR
SC_DIR = '%s/sensor_configurations' % DATA_DIR
SCENARIO_DIR_NEW = '%s/scenarios' % DATA_DIR
SOLVER_DIR = '%s/solvers' % DATA_DIR

SCENARIO_DIR = '{0}/traffic-estimation/comparison/hadoop/input'
RESULT_DIR = '{0}/traffic-estimation/comparison/output'

# Replace with repository homes
REPOSITORIES = { 
        'synthetic': '{0}/traffic-estimation/synthetic_traffic',
        'LS': '{0}/traffic-estimation/BSC_NNLS',
        'BI': '{0}/traffic-estimation/bayesian',
        'KSP': '{0}/traffic-estimation/synthetic_traffic/grid_networks/YenKSP' }

import os
for (k,v) in REPOSITORIES.iteritems():
    # add to path after current dir
    os.sys.path.insert(1,v)
'''
f = open(base + "/comparison/src/config.py", 'w')
f.write(config.replace("{0}", base))

PARENT = '{0}/traffic-estimation'

import logging
logging.basicConfig(level=logging.WARN)

import os
for (k,v) in REPOSITORIES.iteritems():
    # add to path after current dir
    os.sys.path.insert(1,v)
os.sys.path.insert(1,PARENT)
